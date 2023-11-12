FROM nvcr.io/nvidia/tritonserver:21.10-py3
RUN pip install --upgrade pip
RUN pip install torch==2.0.1 transformers==4.33.1 pathlib loguru