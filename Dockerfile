FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
FROM python:3.10

ARG MODEL
WORKDIR bloom_lora_finetune

RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs


RUN git clone https://huggingface.co/bigscience/$MODEL

COPY data/train_tilde.json ./data/train_tilde.json
COPY train_config.json .
COPY finetune.py .



CMD python finetune.py

