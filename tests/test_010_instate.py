"""Tests for the public Instate API."""

from typing import Any

import pandas as pd
import pytest

import instate

NAMES = ["sood", "chintalapati", "sharma"]
STATES = ["Delhi", "Punjab", "Karnataka"]


def test_get_state_distribution_list() -> None:
    """Electoral-roll lookup accepts a list."""
    result = instate.get_state_distribution(NAMES)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "name" in result.columns
    assert len(result.columns) > 31


def test_get_state_distribution_dataframe() -> None:
    """Electoral-roll lookup accepts a DataFrame."""
    frame = pd.DataFrame({"lastname": NAMES})
    result = instate.get_state_distribution(frame, "lastname")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "lastname" in result.columns
    assert len(result.columns) > 31


def test_get_state_distribution_v2_new_states() -> None:
    """The lookup includes the three states omitted from version 1."""
    result = instate.get_state_distribution(["sood", "nair"])
    for state in ["Himachal Pradesh", "Tamil Nadu", "West Bengal"]:
        assert state in result.columns

    sood = result[result["name"] == "sood"].iloc[0]
    assert sood["Punjab"] > 0


def test_get_state_distribution_preserves_every_input_row() -> None:
    """Lookup preserves order, duplicates, short names, and missing names."""
    names = ["sood", "li", "sood", None, "unknownsurnamezz"]
    frame = pd.DataFrame({"lastname": names})

    result = instate.get_state_distribution(frame, "lastname")

    assert len(result) == len(names)
    assert result["lastname"].tolist()[:3] == names[:3]
    assert pd.isna(result.loc[1, "Punjab"])
    assert pd.isna(result.loc[3, "Punjab"])
    assert result.loc[0, "Punjab"] == result.loc[2, "Punjab"]


def test_predict_state() -> None:
    """BiLSTM state prediction returns the requested number of states."""
    result = instate.predict_state(NAMES, top_k=3)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "predicted_states" in result.columns
    predictions = result["predicted_states"].iloc[0]
    assert isinstance(predictions, list)
    assert len(predictions) == 3


def test_predict_state_batched_equivalence() -> None:
    """Batched state inference matches single-name inference and keeps order."""
    names = ["sood", "patil", "nair", "ab", "", "yadav"]
    predictions = list(instate.predict_state(names)["predicted_states"])

    assert len(predictions) == len(names)
    assert predictions[3] == []
    assert predictions[4] == []
    assert len(predictions[0]) == 3
    single = instate.predict_state(["sood"])["predicted_states"].iloc[0]
    assert predictions[0] == single


def test_predict_language_lstm() -> None:
    """BiLSTM language prediction returns the requested number of languages."""
    result = instate.predict_language(NAMES, model="lstm", top_k=3)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "predicted_languages" in result.columns
    predictions = result["predicted_languages"].iloc[0]
    assert isinstance(predictions, list)
    assert len(predictions) == 3


def test_predict_language_lstm_batched() -> None:
    """Batched language inference preserves order and handles short names."""
    names = ["reddy", "menon", "gill", "ab", "", "das"]
    predictions = list(
        instate.predict_language(names, model="lstm")["predicted_languages"]
    )

    assert len(predictions) == len(names)
    assert predictions[3] == []
    assert predictions[4] == []
    assert len(predictions[0]) == 3
    single = instate.predict_language(["reddy"], model="lstm")[
        "predicted_languages"
    ].iloc[0]
    assert predictions[0] == single


def test_predict_language_knn() -> None:
    """KNN language prediction returns one language per name."""
    result = instate.predict_language(NAMES, model="knn")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "predicted_languages" in result.columns
    assert isinstance(result["predicted_languages"].iloc[0], str)


def test_get_state_languages() -> None:
    """State lookup returns official languages."""
    result = instate.get_state_languages(STATES)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "state" in result.columns
    assert "official_languages" in result.columns


def test_list_available_states() -> None:
    """The state list contains only the 34 state columns."""
    states = instate.list_available_states()

    assert isinstance(states, list)
    assert len(states) == 34
    assert "Delhi" in states
    assert "total_n" not in states


@pytest.mark.parametrize("model", ["invalid", "gru"])
def test_predict_state_rejects_invalid_model(model: str) -> None:
    """State prediction rejects unsupported model names."""
    with pytest.raises(ValueError, match="not supported"):
        instate.predict_state(NAMES, model=model)


def test_predict_language_rejects_invalid_model() -> None:
    """Language prediction rejects unsupported model names."""
    with pytest.raises(ValueError, match="not supported"):
        instate.predict_language(NAMES, model="invalid")


@pytest.mark.parametrize("top_k", [0, -1, 35])
def test_predict_state_rejects_invalid_top_k(top_k: int) -> None:
    """State prediction validates the requested result count."""
    with pytest.raises(ValueError, match="between 1 and 34"):
        instate.predict_state(NAMES, top_k=top_k)


@pytest.mark.parametrize("top_k", [0, -1, 38])
def test_predict_language_rejects_invalid_top_k(top_k: int) -> None:
    """Language prediction validates the requested result count."""
    with pytest.raises(ValueError, match="between 1 and 37"):
        instate.predict_language(NAMES, top_k=top_k)


@pytest.mark.parametrize("top_k", [True, 1.5, "3"])
def test_predict_state_rejects_noninteger_top_k(top_k: Any) -> None:
    """State prediction rejects noninteger result counts."""
    with pytest.raises(TypeError, match="must be an integer"):
        instate.predict_state(NAMES, top_k=top_k)


def test_empty_input() -> None:
    """Lookup and prediction preserve empty inputs."""
    assert instate.get_state_distribution([]).empty
    assert instate.predict_state([]).empty
