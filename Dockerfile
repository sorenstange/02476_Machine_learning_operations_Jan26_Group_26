FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt update && \
    apt install --no-install-recommends -y \
        python3 \
        python3-pip \
        git \
        build-essential && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install dvc[gcs]

# Copy code and DVC metadata
COPY src/ src/
COPY configs/ configs/
COPY .dvc/ .dvc/
COPY data.dvc .

# Default command
ENTRYPOINT ["bash", "-c"]
CMD ["dvc pull && python src/train.py experiment=cnn"]
