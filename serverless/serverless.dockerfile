FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

USER root

# install git so that `huggingface-cli` can call it
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install ninja packaging
RUN pip install mlflow protobuf huggingface_hub wandb transformers accelerate peft datasets sentencepiece liger-kernel
RUN pip install pytorch-ignite
RUN pip install bitsandbytes
RUN pip install optuna
RUN pip install unsloth
RUN pip install trl
RUN pip install --upgrade transformers
RUN pip install --no-build-isolation axolotl
RUN pip install runpod
RUN pip install --no-build-isolation flash-attn
RUN pip install deepspeed

WORKDIR /workspace
RUN mkdir -p /workspace/configs /workspace/outputs /workspace/data /workspace/input_data /workspace/training

ENV CONFIG_DIR="/workspace/configs"
ENV OUTPUT_DIR="/workspace/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233bcef0e60addba6

RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config

    
#### ENV Setup
# ───────────────────────────────────────────────────────────────────────────
# 1. Disable Hugging Face tokenizer threads—they conflict with DataLoader
ENV TOKENIZERS_PARALLELISM=false

# 2. Pin CPU-side BLAS threads to 1 to avoid oversubscription
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# 3. Enable cuDNN autotuner for best conv performance
ENV CUDNN_BENCHMARK=1
ENV CUDNN_DETERMINISTIC=0

# 4. Avoid tiny-block fragmentation in PyTorch’s CUDA allocator
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 6.NCCL tuning for multi-GPU efficiency
ENV TORCH_NCCL_BLOCKING_WAIT=1
# ───────────────────────────────────────────────────────────────────────────

COPY serverless/runpod_handler.py /workspace/configs
COPY serverless/serverless_config_handler.py /workspace/configs
COPY serverless/base.yml /workspace/configs
COPY serverless/base_testing.yml /workspace/configs
COPY serverless/accelerate.yaml /workspace/configs

COPY serverless/hpo_optuna.py /workspace/training
COPY serverless/train.py /workspace/training
COPY serverless/train_dpo.py /workspace/training
COPY serverless/train_grpo.py /workspace/training

CMD echo 'Preparing logging...' && \
    echo "Attempting to log in to Hugging Face" && \
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential && \
    echo "Attempting to log in to W&B" && \
    wandb login "$WANDB_TOKEN" && \
    python -u /workspace/configs/runpod_handler.py
# save to phoenixbeaudry/gradients-miner:serverless