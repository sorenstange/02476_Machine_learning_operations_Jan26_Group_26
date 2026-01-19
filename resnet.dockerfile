FROM  nvcr.io/nvidia/pytorch:22.07-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy code and DVC metadata
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

# Default command
ENTRYPOINT ["python", "-u", "src/train.py", "experiment=resnet"]