"""
Neural network predictions for names not in electoral rolls.

Functions for predicting states and languages using trained models.
"""

from __future__ import annotations

from functools import partial

import pandas as pd
import torch
from Levenshtein import distance


def predict_state(
    names: pd.DataFrame | list[str],
    name_column: str | None = None,
    top_k: int = 3,
    model: str = "lstm",
) -> pd.DataFrame:
    """Predict most likely Indian states for given names using a neural network.

    Uses a trained character-level BiLSTM to predict which Indian states a person with
    the given lastname is most likely to be from. This is useful for names not found in
    the electoral rolls data.

    Args:
        names: DataFrame containing names or list of name strings.
            Names are automatically cleaned (lowercase, stripped).
        name_column: If names is a DataFrame, the column containing names.
        top_k: Number of top states to return (default: 3).
        model: Model to use for prediction. Only "lstm" is supported (the legacy
            "gru" was retired in v1.2.0).

    Returns:
        DataFrame with name and predicted_states columns.
        predicted_states contains a list of top_k state names.

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

    if model != "lstm":
        raise ValueError(
            f"Model '{model}' not supported (the GRU was retired in v1.2.0). Use 'lstm'."
        )

    # Prepare DataFrame
    df = prepare_name_dataframe(names, name_column)
    name_col = df.columns[0]

    # Load model
    net = load_state_lstm_model()

    # Encode all names; names too short / with no in-vocab chars predict [] (kept in order).
    predictions: list[list[str]] = [[] for _ in range(len(df))]
    valid_rows: list[int] = []
    valid_enc: list[list[int]] = []
    for row, name in enumerate(df[name_col]):
        cleaned = clean_name(name)
        encoded = encode_name(cleaned) if cleaned and len(cleaned) >= 3 else []
        if encoded:
            valid_rows.append(row)
            valid_enc.append(encoded)

    # Batched inference over the valid names (the model masks PAD via pack_padded_sequence,
    # so this is numerically identical to per-name inference).
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
        top_k: Number of top languages to return (default: 3).
            Note: KNN method returns only the single best match.
        model: Prediction method - "lstm" (neural) or "knn" (lookup).

    Returns:
        DataFrame with name and predicted_languages columns.
        For LSTM: predicted_languages contains list of top_k languages.
        For KNN: predicted_languages contains single best language.

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
    """
    Internal function for LSTM language prediction.
    """
    from ._utils import clean_name, load_lstm_model

    model, lstm_data = load_lstm_model()
    char2idx = lstm_data["char2idx"]  # type: ignore[assignment]
    idx2lang = lstm_data["idx2lang"]  # type: ignore[assignment]
    device = lstm_data["device"]  # type: ignore[assignment]

    predictions: list[list[str]] = []

    for name in names:
        cleaned = clean_name(name)
        if not cleaned or len(cleaned) < 3:
            predictions.append([])
            continue

        # Convert name to indices
        try:
            name_indices: list[int] = [char2idx.get(char, 0) for char in cleaned]  # type: ignore[attr-defined]
        except Exception:
            predictions.append([])
            continue

        # Prepare tensor
        with torch.no_grad():
            name_tensor = torch.tensor(name_indices, dtype=torch.long).unsqueeze(0)
            name_tensor = name_tensor.to(device)  # type: ignore[arg-type]
            lengths = torch.tensor([len(cleaned)], dtype=torch.long)

            # Get predictions for top 3 language outputs
            out1, out2, out3 = model(name_tensor, lengths)

            # Get top predictions from each output
            pred_first = torch.argmax(out1, dim=1)
            pred_second = torch.argmax(out2, dim=1)
            pred_third = torch.argmax(out3, dim=1)

            # Ensure unique predictions
            if pred_second == pred_first:
                pred_second = torch.topk(out2, k=2, dim=1)[1][0][1]
            if pred_third == pred_first or pred_third == pred_second:
                pred_third = torch.topk(out3, k=3, dim=1)[1][0][2]

            # Convert to language names
            langs: list[str] = [
                idx2lang[pred_first.item()],  # type: ignore[index]
                idx2lang[pred_second.item()],  # type: ignore[index]
                idx2lang[pred_third.item()],  # type: ignore[index]
            ]

            # Return only top_k languages
            predictions.append(langs[:top_k])

    return predictions


def _predict_language_knn(names: pd.Series) -> list[str]:
    """
    Internal function for KNN language lookup.
    """
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
