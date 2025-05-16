FROM python:3

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install build-essential libgtk-3-dev libwebkit2gtk-4.0-dev -y
RUN apt-get update && apt-get install -yq libgconf-2-4
RUN apt-get install libsoup2.4 -y
RUN apt-get install gir1.2-javascriptcoregtk-4.0 -y
RUN apt-get install libjavascriptcoregtk-4.0-dev -y

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
    mkdir /var/run/sshd

CMD ["/usr/sbin/sshd", "-D"]


# Set Env Variables
# WANDB_TOKEN
# HUGGINGFACE_USERNAME
# HUGGINGFACE_TOKEN 
# RUNPOD_API_KEY
# WALLET_NAME=default
# HOTKEY_NAME=default
# SUBTENSOR_NETWORK=finney
