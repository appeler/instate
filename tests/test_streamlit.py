"""Tests for the optional Streamlit interface."""

import runpy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1]


def test_deployment_installs_the_project_streamlit_extra() -> None:
    """The cloud deployment uses package metadata as its dependency source."""
    requirements = PROJECT_ROOT / "streamlit" / "requirements.txt"

    assert requirements.read_text().strip() == ".[streamlit]"


def test_app_uses_current_public_api() -> None:
    """The app exposes the maintained lookup and prediction functions."""
    namespace = runpy.run_path(PROJECT_ROOT / "streamlit" / "streamlit_app.py")

    assert namespace["FUNCTIONS"] == {
        "Electoral-roll state distribution": (
            namespace["instate"].get_state_distribution
        ),
        "BiLSTM state prediction": namespace["instate"].predict_state,
    }


def test_download_file_builds_csv_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app exposes results through Streamlit's download control."""
    namespace = runpy.run_path(PROJECT_ROOT / "streamlit" / "streamlit_app.py")
    download_button = Mock()
    monkeypatch.setattr(namespace["st"], "download_button", download_button)

    namespace["download_file"](pd.DataFrame({"name": ["sood"]}))

    assert download_button.call_args.args == ("Download results", "name\nsood\n")
    assert download_button.call_args.kwargs == {
        "file_name": "results.csv",
        "mime": "text/csv",
    }
