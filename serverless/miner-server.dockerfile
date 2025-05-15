FROM ubuntu:devel

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y python3
RUN apt-get install -y python3-pip

RUN pip install --upgrade pip setuptools wheel ninja packaging

WORKDIR /workspace

RUN git clone https://github.com/PhoenixBeaudry/Gradients.git
RUN cd Gradients
RUN git checkout serverless

RUN pip install -e .
RUN pip install runpod
RUN pip install rq

