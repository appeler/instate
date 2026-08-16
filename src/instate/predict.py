"""Neural network predictions for names not in electoral rolls.

Functions for predicting states and languages using trained models.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
from Levenshtein import distance

if TYPE_CHECKING:
    import pandas as pd


def predict_state(
    names: pd.DataFrame | list[str],
    name_column: str | None = None,
    top_k: int = 3,
    model: str = "lstm",
) -> pd.DataFrame:
    """Predict most likely Indian states for given names using a neural network.

    Uses a trained character-level BiLSTM to predict which Indian states a
    person with the given lastname is most likely to be from. This is useful
    for names not found in the electoral rolls data.

    Args:
        names: DataFrame containing names or list of name strings.
            Names are automatically cleaned (lowercase, stripped).
        name_column: If names is a DataFrame, the column containing names.
            Required for DataFrame input.
        top_k: Number of top states to return (default: 3).
        model: Model to use for prediction. Only "lstm" is supported (the legacy
            "gru" was retired in v1.2.0).

    Returns:
        DataFrame with name and predicted_states columns.
        predicted_states contains a list of top_k state names.

    Raises:
        TypeError: If ``top_k`` is not an integer.
        ValueError: If ``top_k`` or ``model`` is invalid.

    Examples:
        >>> names = ["dhingra", "sood", "gowda"]
        >>> result = predict_state(names, top_k=3)
        >>> result["predicted_states"][0]
        ['Delhi', 'Punjab', 'Haryana']

        >>> df = pd.DataFrame({"lastname": ["sharma", "patel"]})
        >>> result = predict_state(df, "lastname", top_k=2)
        >>> len(result["predicted_states"][0])
        2
    """
    from ._utils import clean_name, load_state_lstm_model, prepare_name_dataframe
    from .constants import GT_KEYS
    from .nnets import encode_name, pad_encoded

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if not 1 <= top_k <= len(GT_KEYS):
        raise ValueError(
            f"top_k must be between 1 and {len(GT_KEYS)} for state prediction"
        )

    if model != "lstm":
        raise ValueError(
            f"Model '{model}' not supported. The GRU was retired in v1.2.0; use 'lstm'."
        )

    # Prepare DataFrame
    df = prepare_name_dataframe(names, name_column)
    name_col = df.columns[0]

    # Load model
    net = load_state_lstm_model()

    # Names shorter than three characters or without known characters predict [].
    predictions: list[list[str]] = [[] for _ in range(len(df))]
    valid_rows: list[int] = []
    valid_enc: list[list[int]] = []
    for row, name in enumerate(df[name_col]):
        encoded = encode_name(clean_name(name))
        if len(encoded) >= 3:
            valid_rows.append(row)
            valid_enc.append(encoded)

    # Padding is masked, so batched and per-name inference are numerically identical.
    batch_size = 1024
    for start in range(0, len(valid_enc), batch_size):
        rows = valid_rows[start : start + batch_size]
        x, lengths = pad_encoded(valid_enc[start : start + batch_size])
        with torch.no_grad():
            top = net(x, lengths).topk(top_k, dim=1).indices.tolist()
        for row, idxs in zip(rows, top, strict=True):
            predictions[row] = [GT_KEYS[i] for i in idxs]

    # Add predictions to DataFrame
    result = df.copy()
    result["predicted_states"] = predictions

    return result


def predict_language(
    names: pd.DataFrame | list[str],
    name_column: str | None = None,
    top_k: int = 3,
    model: str = "lstm",
) -> pd.DataFrame:
    """Predict most likely languages for given names.

    Two methods available:
    - "lstm": Neural network prediction using trained LSTM model
    - "knn": K-nearest neighbor lookup in language database

    Args:
        names: DataFrame containing names or list of name strings.
        name_column: If names is a DataFrame, the column containing names.
            Required for DataFrame input.
        top_k: Number of top languages to return (default: 3).
            Note: KNN method returns only the single best match.
        model: Prediction method - "lstm" (neural) or "knn" (lookup).

    Returns:
        DataFrame with name and predicted_languages columns.
        For LSTM: predicted_languages contains list of top_k languages.
        For KNN: predicted_languages contains single best language.

    Raises:
        TypeError: If ``top_k`` is not an integer for LSTM prediction.
        ValueError: If ``top_k`` or ``model`` is invalid.

    Examples:
        >>> names = ["sood", "chintalapati"]
        >>> result = predict_language(names, model="lstm")
        >>> result["predicted_languages"][0]
        ['hindi', 'punjabi', 'urdu']

        >>> result_knn = predict_language(names, model="knn")
        >>> result_knn["predicted_languages"][0]
        'hindi'

        >>> df = pd.DataFrame({"name": ["patel", "sharma"]})
        >>> result = predict_language(df, "name", model="lstm", top_k=2)
        >>> len(result["predicted_languages"][0])
        2
    """
    from ._utils import prepare_name_dataframe

    # Prepare DataFrame
    df = prepare_name_dataframe(names, name_column)
    name_col = df.columns[0]

    if model == "lstm":
        from .constants import IDX_TO_LANG

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if not 1 <= top_k <= len(IDX_TO_LANG):
            raise ValueError(
                f"top_k must be between 1 and {len(IDX_TO_LANG)} for language "
                "prediction"
            )
        predictions = _predict_language_lstm(df[name_col], top_k)
    elif model == "knn":
        predictions = _predict_language_knn(df[name_col])
    else:
        raise ValueError(f"Model '{model}' not supported. Use 'lstm' or 'knn'.")

    # Add predictions to DataFrame
    result = df.copy()
    result["predicted_languages"] = predictions

    return result


def _predict_language_lstm(names: pd.Series, top_k: int = 3) -> list[list[str]]:
    """Predict languages in batches with the character-level BiLSTM."""
    from ._utils import clean_name, load_language_lstm_model
    from .constants import IDX_TO_LANG
    from .nnets import encode_name, pad_encoded

    net = load_language_lstm_model()

    predictions: list[list[str]] = [[] for _ in range(len(names))]
    valid_rows: list[int] = []
    valid_enc: list[list[int]] = []
    for row, name in enumerate(names):
        encoded = encode_name(clean_name(name))
        if len(encoded) >= 3:
            valid_rows.append(row)
            valid_enc.append(encoded)

    batch_size = 1024
    for start in range(0, len(valid_enc), batch_size):
        rows = valid_rows[start : start + batch_size]
        x, lengths = pad_encoded(valid_enc[start : start + batch_size])
        with torch.no_grad():
            top = net(x, lengths).topk(top_k, dim=1).indices.tolist()
        for row, idxs in zip(rows, top, strict=True):
            predictions[row] = [IDX_TO_LANG[i] for i in idxs]

    return predictions


def _predict_language_knn(names: pd.Series) -> list[str]:
    """Look up languages using the nearest surnames."""
    from ._utils import clean_name, load_language_lookup_data

    lang_data = load_language_lookup_data()
    lang_cols = lang_data.columns[1:]  # Skip lastname column

    predictions: list[str] = []

    for name in names:
        cleaned = clean_name(name)
        if not cleaned or len(cleaned) < 3:
            predictions.append("")
            continue

        # Calculate edit distance to all names in database
        # Use partial to avoid lambda scope issue
        distances = lang_data["last_name"].apply(partial(distance, cleaned))  # type: ignore[reportUnknownMemberType]

        # Get top 3 nearest names
        nearest_indices = distances.nsmallest(3).index

        # Sum language scores for nearest names and get max
        lang_scores = lang_data.loc[nearest_indices, lang_cols].sum()  # type: ignore[reportUnknownMemberType]
        best_lang = lang_scores.idxmax()  # type: ignore[reportUnknownMemberType]

        predictions.append(str(best_lang))

    return predictions
