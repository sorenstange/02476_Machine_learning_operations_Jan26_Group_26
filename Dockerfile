FROM  nvcr.io/nvidia/pytorch:22.07-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
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
