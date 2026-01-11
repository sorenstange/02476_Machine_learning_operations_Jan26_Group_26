#!/usr/bin/env bash
set -euo pipefail

# Change to workspace
cd /workspace || exit 1

# Set safe default for W&B: offline unless explicitly provided
export WANDB_MODE=${WANDB_MODE:-offline}

echo "entrypoint: WANDB_MODE=${WANDB_MODE}"

echo "entrypoint: checking dvc configuration..."
if command -v dvc >/dev/null 2>&1; then
  # Only attempt dvc pull if this looks like a repo with git metadata
  if [ -d .git ] || [ -f .dvc/config ]; then
    # If there are no remotes configured, skip pull
    if dvc remote list | grep -q .; then
      echo "entrypoint: attempting dvc pull (configured remote detected)..."
      if dvc pull --verbose; then
        echo "entrypoint: dvc pull succeeded"
      else
        echo "entrypoint: dvc pull failed — continuing without data (mount data or check credentials)" >&2
      fi
    else
      echo "entrypoint: no DVC remote configured; skipping dvc pull"
    fi
  else
    echo "entrypoint: not a git repo and no .dvc present; skipping dvc pull"
  fi
else
  echo "entrypoint: dvc not found in image; skipping dvc pull" >&2
fi

echo "entrypoint: launching training (python3.11 -m src.train)"
exec python3.11 -m src.train "$@"
