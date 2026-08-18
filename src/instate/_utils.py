"""Internal data preparation and model-loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

_CACHE: dict[str, object] = {}


def prepare_name_dataframe(
    names: pd.DataFrame | list[str], name_column: str | None = None
) -> pd.DataFrame:
    """Convert supported input into a DataFrame with the name column first.

    Args:
        names: DataFrame or list of names.
        name_column: Name column when ``names`` is a DataFrame.

    Returns:
        A copy of the input with its name column first.

    Raises:
        ValueError: If a DataFrame has no columns or the requested column is absent.
    """
    if isinstance(names, list):
        return pd.DataFrame({"name": names})
    if names.empty and len(names.columns) == 0:
        raise ValueError("Input DataFrame must contain a name column")

    df = names.copy()
    if name_column is None:
        raise ValueError("name_column must be specified for DataFrame input")

    if name_column not in df.columns:
        raise ValueError(f"Name column '{name_column}' does not exist")
    if name_column != df.columns[0]:
        df = df[
            [name_column, *[column for column in df.columns if column != name_column]]
        ]
    return cast("pd.DataFrame", df)


def prepare_model_input(name: Any) -> tuple[str, list[int], str]:
    """Prepare a surname and report whether the current model can use it.

    Args:
        name: Input surname value.

    Returns:
        The supported-character string, its encoded values, and a stable status.
    """
    from .constants import CHAR_TO_IDX
    from .nnets import canonicalize_name

    if not isinstance(name, str) or not name.strip():
        return "", [], "abstained_empty_or_missing"

    lowered = name.lower()
    supported = canonicalize_name(name)
    unsupported_removed = any(character not in CHAR_TO_IDX for character in lowered)
    if len(supported) < 3:
        reason = (
            "abstained_unsupported_characters"
            if unsupported_removed
            else "abstained_too_short"
        )
        return supported, [], reason

    status = (
        "predicted_unsupported_characters_removed"
        if unsupported_removed
        else "predicted"
    )
    return supported, [CHAR_TO_IDX[character] for character in supported], status


def clean_name(name: Any) -> str:
    """Normalize a name for matching and model input.

    Args:
        name: Input value.

    Returns:
        Lowercase alphabetic characters, or an empty string for missing or
        non-string input.
    """
    if not isinstance(name, str):
        return ""
    return "".join(
        character for character in name.strip().lower() if character.isalpha()
    )


def load_electoral_data() -> pd.DataFrame:
    """Load the bundled, versioned v2 electoral-roll distribution.

    Returns:
        Electoral data with an internal lastname join key.
    """
    if "electoral_v2" not in _CACHE:
        data_dir = Path(__file__).parent / "data"
        path = data_dir / "instate_unique_ln_state_prop_v2.parquet"
        data = pd.read_parquet(path)
        data.rename(columns={"last_name": "__last_name"}, inplace=True)
        _CACHE["electoral_v2"] = data
    return _CACHE["electoral_v2"]  # type: ignore[return-value]


def load_state_lstm_model() -> torch.nn.Module:
    """Load the pinned character-BiLSTM state model.

    Returns:
        Model in evaluation mode.
    """
    if "state_lstm_model" not in _CACHE:
        from .constants import (
            GT_KEYS,
            STATE_LSTM_DROPOUT,
            STATE_LSTM_EMB,
            STATE_LSTM_HIDDEN,
            STATE_LSTM_LAYERS,
            VOCAB_SIZE,
        )
        from .nnets import StateLSTM

        model = StateLSTM(
            VOCAB_SIZE,
            len(GT_KEYS),
            STATE_LSTM_EMB,
            STATE_LSTM_HIDDEN,
            STATE_LSTM_LAYERS,
            STATE_LSTM_DROPOUT,
        )
        from ._resources import resolve_model

        model_file = resolve_model("instate_state_lstm.pt")
        model.load_state_dict(
            torch.load(model_file, map_location="cpu", weights_only=True)
        )
        model.eval()
        _CACHE["state_lstm_model"] = model
    return _CACHE["state_lstm_model"]  # type: ignore[return-value]


def load_language_lstm_model() -> torch.nn.Module:
    """Load the pinned character-BiLSTM language model.

    Returns:
        Model in evaluation mode.
    """
    if "language_lstm_model" not in _CACHE:
        from .constants import (
            LANG_LSTM_DROPOUT,
            LANG_LSTM_EMB,
            LANG_LSTM_HIDDEN,
            LANG_LSTM_LAYERS,
            NUM_LANGUAGES,
            VOCAB_SIZE,
        )
        from .nnets import CharBiLSTM

        model = CharBiLSTM(
            VOCAB_SIZE,
            NUM_LANGUAGES,
            LANG_LSTM_EMB,
            LANG_LSTM_HIDDEN,
            LANG_LSTM_LAYERS,
            LANG_LSTM_DROPOUT,
        )
        from ._resources import resolve_model

        model_file = resolve_model("instate_lang_lstm.pt")
        model.load_state_dict(
            torch.load(model_file, map_location="cpu", weights_only=True)
        )
        model.eval()
        _CACHE["language_lstm_model"] = model
    return _CACHE["language_lstm_model"]  # type: ignore[return-value]


def load_language_lookup_data() -> pd.DataFrame:
    """Load the bundled KNN language lookup table.

    Returns:
        Lastname-to-language scores.

    """
    if "lang_lookup" not in _CACHE:
        path = Path(__file__).parent / "data" / "lastname_langs_india.parquet"
        _CACHE["lang_lookup"] = pd.read_parquet(path)
    return _CACHE["lang_lookup"]  # type: ignore[return-value]
