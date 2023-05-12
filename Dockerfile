FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
FROM python:3.10
RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data/en_es_dataset.json data/en_es_dataset.json
COPY train_config.json .
COPY finetune.py .
COPY pyproject.toml .



CMD python finetune.py

