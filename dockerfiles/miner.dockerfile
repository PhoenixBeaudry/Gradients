FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

USER root

# install git so that `huggingface-cli` can call it
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install ninja packaging
RUN pip install mlflow protobuf huggingface_hub wandb transformers accelerate peft trl datasets sentencepiece liger-kernel
RUN pip install flash-attn --no-build-isolation
RUN pip install pytorch-ignite
RUN pip install bitsandbytes
RUN pip install optuna
RUN pip install --no-build-isolation axolotl
RUN pip install deepspeed
RUN pip install --upgrade transformers
RUN pip install unsloth
WORKDIR /workspace
RUN mkdir -p /workspace/configs /workspace/outputs /workspace/data /workspace/input_data

ENV CONFIG_DIR="/workspace/configs"
ENV OUTPUT_DIR="/workspace/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233bcef0e60addba6

RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config

ENV FLASH_ATTN_FORCE=1
ENV OMP_NUM_THREADS=4
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD echo 'Preparing data...' && \
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \
    echo "Attempting to log in to Hugging Face" && \
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential; \
    else \
    echo "HUGGINGFACE_TOKEN is not set. Skipping Hugging Face login."; \
    fi && \
    if [ -n "$WANDB_TOKEN" ]; then \
    echo "Attempting to log in to W&B" && \
    wandb login "$WANDB_TOKEN"; \
    else \
    echo "WANDB_TOKEN is not set. Skipping W&B login."; \
    fi && \
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/data/${DATASET_FILENAME}; \
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/${DATASET_FILENAME}; \
    fi && \
    echo 'Starting training command' && \
    python /workspace/training/hpo_optuna.py \
  --config           ${CONFIG_DIR}/${JOB_ID}.yml \
  --accelerate_yaml  /workspace/configs/accelerate.yaml \


# save to phoenixbeaudry/gradients-miner:unsloth