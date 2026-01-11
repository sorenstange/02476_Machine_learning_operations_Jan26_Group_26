# CUDA training image for the repository
# Uses a PyTorch image matching torch==2.6.0 from requirements.txt
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG USER=mluser
ARG UID=1000
RUN useradd -m -u ${UID} ${USER} || true

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install Python and build deps
COPY requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    git curl ca-certificates build-essential && \
    python3.11 -m pip install --upgrade pip setuptools wheel && \
    # Install CUDA-enabled PyTorch wheels compatible with CUDA 11.8
    python3.11 -m pip install --no-cache-dir "torch==2.6.0+cu118" "torchvision==0.21.0+cu118" -f https://download.pytorch.org/whl/cu118/torch_stable.html || true && \
    # Install remaining requirements (some packages may already be satisfied)
    python3.11 -m pip install --no-cache-dir -r requirements.txt || true && \
    python3.11 -m pip install --no-cache-dir dvc[s3] || true && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /workspace

# Copy entrypoint and make executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Ensure non-root owns the workspace
RUN chown -R ${USER}:${USER} /workspace || true
USER ${USER}

WORKDIR /workspace

# Default command: entrypoint will attempt `dvc pull` then run training. Override at runtime as needed.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
