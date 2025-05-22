FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive

USER root

# System dependencies and performance tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    ccache \
    numactl \
    libnuma-dev \
    infiniband-diags \
    libibverbs-dev \
    ibutils \
    rdma-core \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# Install ML packages with specific versions for compatibility
RUN pip install --no-cache-dir \
    numpy \
    transformers \
    accelerate \
    datasets\
    sentencepiece\
    huggingface_hub \
    wandb \
    peft \
    bitsandbytes \
    safetensors \
    tokenizers

# Install training frameworks
RUN pip install --no-cache-dir \
    trl \
    liger-kernel \
    pytorch-ignite \
    optuna \
    mlflow \
    protobuf

# Install Flash Attention 2 (much faster than flash-attn v1)
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Install DeepSpeed with CPU Adam and other optimizations
ENV DS_BUILD_CPU_ADAM=1
ENV DS_BUILD_FUSED_ADAM=1
ENV DS_BUILD_FUSED_LAMB=1
ENV DS_BUILD_UTILS=1
ENV DS_BUILD_AIO=0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
RUN pip install --no-cache-dir deepspeed

# Install Triton for kernel compilation
RUN pip install --no-cache-dir triton

# Install xFormers for additional memory-efficient attention
RUN pip install --no-cache-dir xformers

# Install Unsloth with specific configuration
ENV UNSLOTH_COMPILE_TORCH=1
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@main"

# Install additional optimizations
RUN pip install --no-cache-dir \
    einops \
    scipy \
    numba \
    packaging \
    toml \
    hf_xet

# RunPod specific
RUN pip install runpod


WORKDIR /workspace
RUN mkdir -p /workspace/configs /workspace/outputs /workspace/data /workspace/input_data /workspace/training

# Environment variables for optimal performance
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8
ENV VECLIB_MAXIMUM_THREADS=8

# CUDA optimizations
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDNN_BENCHMARK=1
ENV CUDNN_DETERMINISTIC=0
ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# PyTorch optimizations
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:16"
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV TORCH_BACKENDS_CUDNN_BENCHMARK=1
ENV TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=1
ENV TORCH_BACKENDS_CUDNN_ALLOW_TF32=1

# NCCL optimizations for multi-GPU
ENV NCCL_DEBUG=ERROR
ENV NCCL_P2P_LEVEL=NVL
ENV NCCL_IB_DISABLE=0
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_TREE_THRESHOLD=0
ENV NCCL_LL_THRESHOLD=0
ENV NCCL_SHM_DISABLE=0
ENV NCCL_NSOCKS_PERTHREAD=4
ENV NCCL_BUFFSIZE=16777216
ENV NCCL_ALGO=Ring
ENV NCCL_PROTO=LL128

# Triton cache
ENV TRITON_CACHE_DIR="/tmp/.triton"

# DeepSpeed optimizations
ENV DS_ACCELERATOR="cuda"
ENV DS_ENV_FILE="/workspace/.ds_env"

# Create DeepSpeed environment file
RUN echo "DS_BUILD_CPU_ADAM=1" > /workspace/.ds_env && \
    echo "DS_BUILD_FUSED_ADAM=1" >> /workspace/.ds_env && \
    echo "DS_BUILD_FUSED_LAMB=1" >> /workspace/.ds_env

# AWS credentials (keep existing)
ENV CONFIG_DIR="/workspace/configs"
ENV OUTPUT_DIR="/workspace/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233bcef0e60addba6

RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config

# Pre-compile Python files for faster startup
RUN python -m compileall /usr/local/lib/python*/site-packages/

# Copy configuration files
COPY serverless/runpod_handler.py /workspace/configs
COPY serverless/serverless_config_handler.py /workspace/configs
COPY serverless/base.yml /workspace/configs
COPY serverless/base_testing.yml /workspace/configs
COPY serverless/accelerate.yaml /workspace/configs

COPY serverless/hpo_optuna.py /workspace/training
COPY serverless/train.py /workspace/training
COPY serverless/train_dpo.py /workspace/training
COPY serverless/train_grpo.py /workspace/training

# Warmup script to pre-compile Triton kernels (optional but recommended)
RUN echo "import torch; torch.cuda.is_available()" | python

CMD echo 'Preparing logging...' && \
    echo "Attempting to log in to Hugging Face" && \
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential && \
    echo "Attempting to log in to W&B" && \
    wandb login "$WANDB_TOKEN" && \
    python -u /workspace/configs/runpod_handler.py