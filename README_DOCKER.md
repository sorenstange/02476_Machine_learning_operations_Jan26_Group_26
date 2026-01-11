## Docker usage (training image)

Build the training image from the project root:

```bash
docker build -t rice-train:latest .
```

quick CPU test (verifies Python and PyTorch import):

```bash
docker run --rm --entrypoint python3.11 rice-train:latest -c "import sys,torch; print('py', sys.version.split()[0]); print('cuda_available=', torch.cuda.is_available())"
```

Recommended run (mount full repo so `.git` and `.dvc` are available; entrypoint will attempt `dvc pull` if configured):

```bash
docker run --rm -it \
  -v "$(pwd)":/workspace:rw \
  -e WANDB_MODE=offline \
  rice-train:latest \
  --config-path /workspace/configs --config-name config
```

If you prefer `dvc pull` on the host then mount only `data/`:

```bash
dvc pull -v
docker run --rm -it \
  -v "$(pwd)/data":/workspace/data:ro \
  -v "$(pwd)/configs":/workspace/configs:ro \
  -e WANDB_MODE=offline \
  rice-train:latest --config-path /workspace/configs --config-name config
```

If you see DataLoader worker/shared-memory errors, increase container shared memory:

```bash
docker run --rm -it --shm-size=1g -v "$(pwd)":/workspace:rw -e WANDB_MODE=offline rice-train:latest --config-path /workspace/configs --config-name config
```
