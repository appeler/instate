"""Resolve versioned model artifacts from local storage or Hugging Face."""

from __future__ import annotations

import hashlib
import os
from importlib.resources import files
from pathlib import Path

HF_REPO = "gojiberries/instate"
HF_REVISION = "304444b4c7effac1cbe1a997f9d4271db44530e6"
MODEL_DIR_ENV = "INSTATE_MODEL_DIR"

# The revision pin fixes which artifacts these are; the per-file hashes catch
# a corrupted or tampered copy the pin alone cannot. INSTATE_MODEL_DIR
# artifacts are exempt so local development can iterate.
ARTIFACT_SHA256 = {
    "instate_state_lstm.pt": (
        "6c2275d2ed5a9cb072abd8dfad44dc4c41459ffd841969451e3fc0035bc825ad"
    ),
    "instate_state_lstm_calibration.json": (
        "4bb7dcbc9a7387f4eedc5f663f8619b26316b6c3ee05014a67479633b3d2702f"
    ),
    "instate_unique_ln_state_prop_v2.parquet": (
        "6411896fa41b2130ee52ab1cb46c69b3a251fc770fc8d0e6d522eab1f5f92270"
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
