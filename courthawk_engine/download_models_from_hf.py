"""
Downloads the model weights from the public HuggingFace repo named by the HF_MODELS_REPO
env var into courthawk_engine/models/.

Dockerfile sets HF_MODELS_REPO=AgentXCross/courthawk-models and runs this script automatically during the
Docker build, so the weights end up in the image without ever being committed to git.
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    repo_id = os.environ["HF_MODELS_REPO"]

    snapshot_download(repo_id = repo_id, local_dir = str(MODELS_DIR))

    print(f"Downloaded models from {repo_id} into {MODELS_DIR}")


if __name__ == "__main__":
    main()
