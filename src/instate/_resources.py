"""Resolve versioned model artifacts from local storage or Hugging Face."""

from __future__ import annotations

import hashlib
import os
from importlib.resources import files
from pathlib import Path

HF_REPO = "gojiberries/instate"
HF_REVISION = "bff80c9c3b828c5f03ec2ae32f77edb7bb31240d"
MODEL_DIR_ENV = "INSTATE_MODEL_DIR"

# The revision pin fixes which artifacts these are; the per-file hashes catch
# a corrupted or tampered copy the pin alone cannot. INSTATE_MODEL_DIR
# artifacts are exempt so local development can iterate.
ARTIFACT_SHA256 = {
    "instate_state_lstm.pt": (
        "e73a4bc1d6d66cec2a8b6261df3f3aba2a646e935513274ef30c7f3f2cbd9c81"
    ),
    "instate_state_lstm_calibration.json": (
        "07117822802c593616c1cb0a58acf2987755f5150342a78e7e2bd8fb1b12e65d"
    ),
    "instate_unique_ln_state_prop_v2.parquet": (
        "055490831cbee38ee1acb167066d90a9ce5f27ed1e1441ac3bd09bcd3e3624f6"
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
