FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt update && \
    apt install --no-install-recommends -y \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        gcc && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (for caching)
COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copy project files
COPY src/ src/
COPY data/ data/
COPY configs/ configs/

# Install your package
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/train.py"]
CMD ["experiment=cnn"]
