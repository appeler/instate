"""Contracts for model resolution from Hugging Face."""

from pathlib import Path
from unittest.mock import patch

import pytest

from instate import _resources
from instate._resources import HF_REPO, HF_REVISION, resolve_model


def test_local_model_override_avoids_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local development model takes precedence over the Hub."""
    model = tmp_path / "instate_state_lstm.pt"
    model.write_bytes(b"weights")
    monkeypatch.setenv("INSTATE_MODEL_DIR", str(tmp_path))

    with patch("huggingface_hub.hf_hub_download") as download:
        assert resolve_model(model.name) == str(model)

    download.assert_not_called()


def test_missing_model_uses_exact_pinned_hub_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader and published repository agree on path and revision."""
    monkeypatch.setenv("INSTATE_MODEL_DIR", str(tmp_path))

    cached = tmp_path / "cache-state.pt"
    cached.write_bytes(b"downloaded weights")
    monkeypatch.setattr(
        "instate._resources.ARTIFACT_SHA256",
        {"instate_state_lstm.pt": _resources._sha256(str(cached))},
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as download:
        assert resolve_model("instate_state_lstm.pt") == str(cached)

    download.assert_called_once_with(
        HF_REPO, "instate_state_lstm.pt", revision=HF_REVISION
    )


def test_resolved_artifact_failing_its_pinned_hash_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupted download or packaged file cannot be used silently."""
    corrupted = tmp_path / "cache-state.pt"
    corrupted.write_bytes(b"not the pinned bytes")
    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(corrupted)),
        pytest.raises(RuntimeError, match="does not match the pinned"),
    ):
        resolve_model("instate_state_lstm.pt")


@pytest.mark.live
def test_pinned_hub_revision_contains_every_model() -> None:
    """The immutable Hub revision contains every artifact the package requests."""
    from huggingface_hub import list_repo_files

    published = set(list_repo_files(HF_REPO, revision=HF_REVISION))
    assert set(_resources.ARTIFACT_SHA256) <= published
