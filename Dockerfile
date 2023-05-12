FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
COPY requirements.txt requirements.txt
RUN pip install -r requierements.txt

COPY finetune.py finetune.py


