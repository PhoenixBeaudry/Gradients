FROM --platform=linux/amd64 runpod/base:0.6.3-cpu

USER root

RUN apt-get update && apt-get install build-essential

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel ninja packaging runpod rq six

RUN pip install bittensor-cli

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /.local/bin

ENV PATH="/.local/bin:$PATH"

RUN git clone https://github.com/PhoenixBeaudry/Gradients.git /Gradients

WORKDIR /Gradients

RUN git checkout serverless

RUN pip install -e .

# Create .1.env with default values as comments
RUN echo -e "WALLET_NAME=default\n HOTKEY_NAME=default\n SUBTENSOR_NETWORK=finney\n NETUID=56" > /Gradients/.1.env



# Set Env Variables
# RUNPOD_API_KEY
# WALLET_NAME=default
# HOTKEY_NAME=default
# SUBTENSOR_NETWORK=finney

