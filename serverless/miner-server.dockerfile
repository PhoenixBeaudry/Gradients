FROM python:3

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel ninja packaging runpod rq bittensor-cli

RUN git clone https://github.com/PhoenixBeaudry/Gradients.git

WORKDIR /Gradients

RUN git checkout miner-docker

RUN pip install -e .

# Set Env Variables
# WANDB_TOKEN
# HUGGINGFACE_USERNAME
# HUGGINGFACE_TOKEN 
# RUNPOD_API_KEY
# WALLET_NAME=default
# HOTKEY_NAME=default
# SUBTENSOR_NETWORK=finney
