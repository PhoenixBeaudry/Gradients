FROM python:3

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel ninja packaging runpod rq

RUN git clone https://github.com/PhoenixBeaudry/Gradients.git

WORKDIR /Gradients

RUN git checkout serverless

RUN pip install -e .

