"""Tests for the public Instate API."""

from typing import Any

import pandas as pd
import pytest

import instate
from instate._utils import load_electoral_data, load_language_lookup_data

NAMES = ["sood", "chintalapati", "sharma"]
STATES = ["Delhi", "Punjab", "Karnataka"]


def test_runtime_tables_have_stable_dtypes() -> None:
    """Packaged Parquet tables preserve their declared logical types."""
    electoral = load_electoral_data()
    languages = load_language_lookup_data()

    assert str(electoral["__last_name"].dtype) == "str"
    assert str(electoral["total_n"].dtype) == "int64"
    assert set(map(str, electoral.drop(columns=["__last_name", "total_n"]).dtypes)) == {
        "float64"
    }
    assert str(languages["last_name"].dtype) == "str"
    assert set(map(str, languages.drop(columns="last_name").dtypes)) == {"float64"}


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


@pytest.mark.parametrize(
    "function",
    [
        instate.get_state_distribution,
        instate.predict_state,
        instate.predict_language,
    ],
)
def test_dataframe_input_requires_name_column(function: Any) -> None:
    """DataFrame APIs never guess which column contains last names."""
    frame = pd.DataFrame({"username": ["unknown"], "lastname": ["sood"]})

    with pytest.raises(
        ValueError, match="name_column must be specified for DataFrame input"
    ):
        function(frame)


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


def test_get_state_distribution_replaces_outputs_and_preserves_index() -> None:
    """Lookup replaces stale output columns without changing the input index."""
    frame = pd.DataFrame(
        {"lastname": ["sood"], "Punjab": [-1.0], "total_n": [-1]},
        index=pd.Index([42], name="row_id"),
    )

    result = instate.get_state_distribution(frame, "lastname")

    assert result.index.equals(frame.index)
    assert result.columns.is_unique
    assert result.loc[42, "Punjab"] == pytest.approx(0.3649627589)
    assert result.loc[42, "total_n"] == 29403


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


def test_prediction_statuses_explain_abstention_and_character_filtering() -> None:
    """All model paths use the same stable input-status vocabulary."""
    names = [None, "ab", "नाम", "éabc", "patel"]
    expected = [
        "abstained_empty_or_missing",
        "abstained_too_short",
        "abstained_unsupported_characters",
        "predicted_unsupported_characters_removed",
        "predicted",
    ]

    state = instate.predict_state(names)
    language_lstm = instate.predict_language(names, model="lstm")
    language_knn = instate.predict_language(names, model="knn")

    assert state["prediction_status"].tolist() == expected
    assert language_lstm["prediction_status"].tolist() == expected
    assert language_knn["prediction_status"].tolist() == expected
    assert state.loc[2, "predicted_states"] == []
    assert language_knn.loc[2, "predicted_languages"] == ""


def test_model_metadata_declares_supported_input_script() -> None:
    """Every inference path documents the alphabet used to train or match it."""
    metadata = instate.get_model_metadata()

    assert set(metadata) == {"state:lstm", "language:lstm", "language:knn"}
    for model in metadata.values():
        assert model == {
            "supported_script": "Latin (ASCII a-z)",
            "supported_characters": "abcdefghijklmnopqrstuvwxyz",
            "minimum_supported_characters": 3,
        }


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


def test_lstm_requires_three_supported_characters() -> None:
    """Unsupported Unicode letters do not count toward the minimum length."""
    assert instate.predict_state(["éab"])["predicted_states"].iloc[0] == []
    assert instate.predict_language(["éab"])["predicted_languages"].iloc[0] == []


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


def test_get_state_languages_supports_pre_union_territories() -> None:
    """Electoral-roll territory names map through the shared language alias."""
    states = ["Dadra and Nagar Haveli", "Daman and Diu"]

    result = instate.get_state_languages(states)

    assert result["state"].tolist() == states
    assert result["official_languages"].tolist() == ["Hindi, English"] * 2


def test_get_state_languages_replaces_outputs_and_preserves_index() -> None:
    """Language lookup replaces stale outputs while retaining source columns."""
    frame = pd.DataFrame(
        {
            "origin": ["Delhi"],
            "state": ["keep me"],
            "official_languages": ["stale"],
        },
        index=pd.Index([42], name="row_id"),
    )

    result = instate.get_state_languages(frame, "origin")

    assert result.index.equals(frame.index)
    assert result.columns.is_unique
    assert result.loc[42, "state"] == "keep me"
    assert result.loc[42, "official_languages"] == "Hindi, English"


def test_get_state_languages_rejects_missing_state_column() -> None:
    """An explicit missing state column raises the documented error type."""
    frame = pd.DataFrame({"state": ["Delhi"]})

    with pytest.raises(ValueError, match="State column 'missing' does not exist"):
        instate.get_state_languages(frame, "missing")


def test_get_state_languages_dataframe_requires_state_column() -> None:
    """State-language lookup never guesses which DataFrame column to use."""
    frame = pd.DataFrame({"state": ["Delhi"]})

    with pytest.raises(
        ValueError, match="state_column must be specified for DataFrame input"
    ):
        instate.get_state_languages(frame)


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
