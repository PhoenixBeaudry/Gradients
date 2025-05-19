FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install build-essential

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel ninja packaging runpod rq

RUN pip install bittensor-cli

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /.local/bin

ENV PATH="/.local/bin:$PATH"

RUN git clone https://github.com/PhoenixBeaudry/Gradients.git

WORKDIR /Gradients

RUN git checkout miner-docker

RUN pip install -e .

RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]


# Set Env Variables
# WANDB_TOKEN
# HUGGINGFACE_USERNAME
# HUGGINGFACE_TOKEN 
# RUNPOD_API_KEY
# WALLET_NAME=default
# HOTKEY_NAME=default
# SUBTENSOR_NETWORK=finney

