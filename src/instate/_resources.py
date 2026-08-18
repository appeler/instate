"""Resolve versioned model artifacts from local storage or Hugging Face."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

HF_REPO = "gojiberries/instate"
HF_REVISION = "cd57ba3123a067ea2b774d6c66b8d7f7483230bd"
MODEL_DIR_ENV = "INSTATE_MODEL_DIR"


def resolve_model(filename: str) -> str:
    """Return a local path for a pinned model artifact.

    Args:
        filename: Filename at the root of the model repository.

    Returns:
        A filesystem path suitable for ``torch.load``.
    """
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    packaged = Path(str(files("instate") / "data" / filename))
    if packaged.is_file():
        return str(packaged)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, filename, revision=HF_REVISION)
