"""Resolve versioned model artifacts from local storage or Hugging Face."""

from __future__ import annotations

import hashlib
import os
from importlib.resources import files
from pathlib import Path

HF_REPO = "gojiberries/instate"
HF_REVISION = "901cc76dc8af03cfe81287a81a196e0752ba1c3e"
MODEL_DIR_ENV = "INSTATE_MODEL_DIR"

# Per-file hashes bind the matching checkpoint, calibration, and lookup.
# INSTATE_MODEL_DIR artifacts are exempt so local development can iterate.
ARTIFACT_SHA256 = {
    "instate_state_lstm.pt": (
        "3d6b762f0b4ac646fd53cc110a3a18c7394a9ec47808d24188279c01aff80ec8"
    ),
    "instate_state_lstm_calibration.json": (
        "c42a748c5e44218853f1d8100ca46ea106154fc4a023c1f9818d3f5e696959a5"
    ),
    "instate_unique_ln_state_prop_v2.parquet": (
        "077dc975f4ea105b5551a512cbea635b3410e8754a763ad94b5ab8e8d7281da1"
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
        A filesystem path suitable for ``torch.load`` or ``read_parquet``.

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
