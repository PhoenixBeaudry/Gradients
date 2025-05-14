FROM winglian/axolotl:main-20250429

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install docker toml

COPY . .

ENV JOB_ID=""
ENV DATASET="chtmp223/suri"
ENV MODELS="FormlessAI/810c4257-bece-43a0-9684-746534b3ab71"
ENV ORIGINAL_MODEL="TinyLlama/TinyLlama_v1.1"
ENV DATASET_TYPE="DpoDatasetType"
ENV FILE_FORMAT="hf"

RUN mkdir /aplp
