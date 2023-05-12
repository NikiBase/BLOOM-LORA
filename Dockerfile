FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
WORKDIR bloom-lora
RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
COPY requirements.txt .
RUN pip install -r requierements.txt

COPY data/en_es_dataset.json data/en_es_dataset.json
COPY train_config.json .
COPY finetune.py .
COPY pyproject.toml .


RUN pip install -e ./

CMD python finetune.py

