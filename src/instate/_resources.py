"""Resolve versioned model artifacts from local storage or Hugging Face."""

from __future__ import annotations

import hashlib
import os
from importlib.resources import files
from pathlib import Path

HF_REPO = "gojiberries/instate"
HF_REVISION = "1146c3a73d7280c68fabc2da7b5436eae2120000"
MODEL_DIR_ENV = "INSTATE_MODEL_DIR"

# Per-file hashes bind the matching checkpoint, calibration, and lookup.
# INSTATE_MODEL_DIR artifacts are exempt so local development can iterate.
ARTIFACT_SHA256 = {
    "instate_state_lstm.safetensors": (
        "0c73a68807bea18ff5472aad3f7fc8b741bcd89b8935a4d0728c36a3d26b9294"
    ),
    "instate_state_lstm_calibration.json": (
        "c59d8dc7e9425ce1cc4b1494bc60d2820b66c7ed4c3758c2e6ce2b8236f339bd"
    ),
    "instate_unique_ln_state_prop_v2.parquet": (
        "eee2adbb6e5803016878ae0ae7f66afadd2a1d01b485928d481def73e64dcad6"
    ),
}


def _sha256(path: str) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: str, filename: str) -> str:
    """Return ``path`` after checking the artifact's pinned hash."""
    expected = ARTIFACT_SHA256[filename]
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{filename}: SHA-256 {actual} does not match the pinned {expected}"
        )
    return path


def resolve_model(filename: str) -> str:
    """Return a local path for a pinned model artifact.

    Resolution order: the ``INSTATE_MODEL_DIR`` override (unverified, for
    development), a file packaged in the wheel, then the pinned Hugging Face
    revision. Packaged and downloaded artifacts must match their pinned
    SHA-256; a mismatch is a ``RuntimeError``.

    Args:
        filename: Filename at the root of the model repository.

    Returns:
        A filesystem path for loading tensors or a Parquet table.

    """
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    packaged = Path(str(files("instate") / "data" / filename))
    if packaged.is_file():
        return _verified(str(packaged), filename)

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(HF_REPO, filename, revision=HF_REVISION)
    return _verified(downloaded, filename)
