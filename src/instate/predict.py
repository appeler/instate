"""Model rankings for names not in electoral-roll lookup tables.

Functions for ranking state and synthetic language targets.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import torch
from Levenshtein import distance  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    import pandas as pd


def predict_state(
    names: pd.DataFrame | list[str],
    name_column: str | None = None,
    top_k: int = 3,
    model: str = "lstm",
) -> pd.DataFrame:
    """Rank Indian state labels learned from processed surname occurrences.

    The character-level BiLSTM targets the distribution of processed surname
    occurrences across included 2017 electoral-roll state records. It does not
    predict an individual's residence or origin.

    Args:
        names: DataFrame containing names or list of name strings.
            Names are automatically cleaned (lowercase, stripped).
        name_column: If names is a DataFrame, the column containing names.
            Required for DataFrame input.
        top_k: Number of top states to return (default: 3).
        model: Model to use for prediction. Only "lstm" is supported (the legacy
            "gru" was retired in v1.2.0).

    Returns:
        DataFrame with name, predicted_states, and prediction_status columns.
        predicted_states contains a list of top_k state names.
        prediction_status explains whether a ranking was produced or why the
        model abstained.

    Raises:
        TypeError: If ``top_k`` is not an integer.
        ValueError: If ``top_k`` or ``model`` is invalid.

    Examples:
        >>> from instate import predict_state
        >>> names = ["dhingra", "sood", "gowda"]
        >>> result = predict_state(names, top_k=3)
        >>> result["predicted_states"][0]
        ['Delhi', 'Uttar Pradesh', 'Haryana']

        >>> import pandas as pd
        >>> df = pd.DataFrame({"lastname": ["sharma", "patel"]})
        >>> result = predict_state(df, "lastname", top_k=2)
        >>> len(result["predicted_states"][0])
        2
    """
    from ._utils import (
        load_state_lstm_model,
        prepare_model_input,
        prepare_name_dataframe,
    )
    from .constants import GT_KEYS
    from .nnets import pad_encoded

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

    # Preserve abstentions in row order and batch only supported inputs.
    predictions: list[list[str]] = [[] for _ in range(len(df))]
    statuses: list[str] = []
    valid_rows: list[int] = []
    valid_enc: list[list[int]] = []
    for row, name in enumerate(df[name_col]):
        _, encoded, status = prepare_model_input(name)
        statuses.append(status)
        if encoded:
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
    result["prediction_status"] = statuses

    return result


def predict_language(
    names: pd.DataFrame | list[str],
    name_column: str | None = None,
    top_k: int = 3,
    model: str = "lstm",
) -> pd.DataFrame:
    """Rank labels from the synthetic language target for given surnames.

    The target mixes ranked state languages with fixed geometric weights and a
    surname's electoral-roll state shares. It is not an observed or official
    language. Two methods are available:
    - "lstm": neural ranking using the trained LSTM model
    - "knn": nearest-surname lookup in the synthetic target table

    Args:
        names: DataFrame containing names or list of name strings.
        name_column: If names is a DataFrame, the column containing names.
            Required for DataFrame input.
        top_k: Number of top languages to return (default: 3).
            Note: KNN method returns only the single best match.
        model: Prediction method - "lstm" (neural) or "knn" (lookup).

    Returns:
        DataFrame with name, predicted_languages, and prediction_status columns.
        For LSTM: predicted_languages contains list of top_k languages.
        For KNN: predicted_languages contains single best language.

    Raises:
        TypeError: If ``top_k`` is not an integer for LSTM prediction.
        ValueError: If ``top_k`` or ``model`` is invalid.

    Examples:
        >>> from instate import predict_language
        >>> names = ["sood", "chintalapati"]
        >>> result = predict_language(names, model="lstm")
        >>> result["predicted_languages"][0]
        ['hindi', 'punjabi', 'urdu']

        >>> result_knn = predict_language(names, model="knn")
        >>> result_knn["predicted_languages"][0]
        'hindi'

        >>> import pandas as pd
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
        name_series = cast("pd.Series", df.loc[:, name_col])
        predictions, statuses = _predict_language_lstm(name_series, top_k)
    elif model == "knn":
        name_series = cast("pd.Series", df.loc[:, name_col])
        predictions, statuses = _predict_language_knn(name_series)
    else:
        raise ValueError(f"Model '{model}' not supported. Use 'lstm' or 'knn'.")

    # Add predictions to DataFrame
    result = df.copy()
    result["predicted_languages"] = predictions
    result["prediction_status"] = statuses

    return result


def _predict_language_lstm(
    names: pd.Series, top_k: int = 3
) -> tuple[list[list[str]], list[str]]:
    """Predict languages in batches with the character-level BiLSTM."""
    from ._utils import load_language_lstm_model, prepare_model_input
    from .constants import IDX_TO_LANG
    from .nnets import pad_encoded

    net = load_language_lstm_model()

    predictions: list[list[str]] = [[] for _ in range(len(names))]
    statuses: list[str] = []
    valid_rows: list[int] = []
    valid_enc: list[list[int]] = []
    for row, name in enumerate(names):
        _, encoded, status = prepare_model_input(name)
        statuses.append(status)
        if encoded:
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

    return predictions, statuses


def _predict_language_knn(names: pd.Series) -> tuple[list[str], list[str]]:
    """Look up synthetic language labels using the nearest surnames."""
    from ._utils import load_language_lookup_data, prepare_model_input

    lang_data = load_language_lookup_data()
    lang_cols = lang_data.columns[1:]  # Skip lastname column

    predictions: list[str] = []
    statuses: list[str] = []
    last_names = cast("pd.Series", lang_data.loc[:, "last_name"])

    for name in names:
        supported, encoded, status = prepare_model_input(name)
        statuses.append(status)
        if not encoded:
            predictions.append("")
            continue

        # Calculate edit distance to all names in database
        # Use partial to avoid lambda scope issue
        distances = cast(
            "pd.Series",
            last_names.apply(partial(distance, supported)),  # type: ignore[reportUnknownMemberType]
        )

        # Get top 3 nearest names
        nearest_indices = distances.nsmallest(3).index

        # Sum language scores for nearest names and get max
        nearest = cast("pd.DataFrame", lang_data.loc[nearest_indices, lang_cols])
        lang_scores = nearest.sum(axis=0)
        best_lang = lang_scores.idxmax()  # type: ignore[reportUnknownMemberType]

        predictions.append(str(best_lang))

    return predictions, statuses


def get_model_metadata() -> dict[str, dict[str, str | int]]:
    """Return input-support metadata for every current inference path."""
    from .constants import MODEL_INPUT_METADATA

    return {name: values.copy() for name, values in MODEL_INPUT_METADATA.items()}
