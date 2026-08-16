"""Electoral rolls based name-to-state lookup.

Functions for looking up state distributions from 2017 Indian electoral rolls data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_state_distribution(
    names: pd.DataFrame | list[str], name_column: str | None = None
) -> pd.DataFrame:
    """Get P(state|lastname) from 2017 Indian electoral rolls.

    This returns the empirical distribution of a lastname across Indian states
    based on the electoral rolls data. This is the Bayes optimal estimate
    given the observed frequencies.

    Args:
        names: DataFrame containing names or list of name strings.
            Names are automatically cleaned (lowercase, stripped).
        name_column: If names is a DataFrame, the column containing names.
            Required for DataFrame input.

    Returns:
        DataFrame with every original row plus 34 state probability columns.
        State columns are named by state (e.g., 'Delhi', 'Punjab').
        Values are proportions (0-1) representing P(state|lastname).

    Examples:
        >>> names = ["dhingra", "sood", "gowda"]
        >>> result = get_state_distribution(names)
        >>> result[["name", "delhi", "punjab", "karnataka"]]

        >>> df = pd.DataFrame({"lastname": ["dhingra", "sood"]})
        >>> result = get_state_distribution(df, "lastname")
        >>> result.columns[:5].tolist()
    """
    from ._utils import clean_name, load_electoral_data, prepare_name_dataframe

    # Convert to DataFrame if needed
    df = prepare_name_dataframe(names, name_column)

    electoral_data = load_electoral_data()
    value_columns = [
        column for column in electoral_data.columns if column != "__last_name"
    ]
    cleaned_names = df.iloc[:, 0].map(clean_name)
    values = electoral_data.set_index("__last_name").reindex(cleaned_names)[
        value_columns
    ]

    result = df.copy()
    for column in value_columns:
        result[column] = values[column].to_numpy()
    return result


def get_state_languages(
    states: pd.DataFrame | list[str], state_column: str | None = None
) -> pd.DataFrame:
    """Map Indian states to their official languages.

    Based on census data, returns the official language(s) for each state.

    Args:
        states: DataFrame containing states or list of state names.
        state_column: If states is a DataFrame, the column containing state names.
            Required for DataFrame input.

    Returns:
        DataFrame with state and official_languages columns.
        If input was DataFrame, adds official_languages column.

    Raises:
        ValueError: If ``state_column`` is missing or invalid for DataFrame input.

    Examples:
        >>> states = ["Delhi", "Punjab", "Karnataka"]
        >>> result = get_state_languages(states)
        >>> result[["state", "official_languages"]]

        >>> df = pd.DataFrame({"state_name": ["Delhi", "Punjab"]})
        >>> result = get_state_languages(df, "state_name")
    """
    # Prepare DataFrame
    if isinstance(states, list):
        df = pd.DataFrame({"state": states})
        state_col = "state"
    else:
        df = states.copy()
        if state_column is None:
            raise ValueError("state_column must be specified for DataFrame input")
        state_col = state_column

    from .constants import STATE_LANGUAGE_ALIASES

    if state_col not in df.columns:
        raise ValueError(f"State column '{state_col}' does not exist")

    state_lang_path = Path(__file__).parent / "data" / "state_to_languages.parquet"
    state_lang_map = pd.read_parquet(state_lang_path)
    value_columns = [column for column in state_lang_map.columns if column != "state"]
    state_keys = df[state_col].replace(STATE_LANGUAGE_ALIASES)
    values = state_lang_map.set_index("state").reindex(state_keys)[value_columns]

    result = df.copy()
    for column in value_columns:
        result[column] = values[column].to_numpy()
    return result


def list_available_states() -> list[str]:
    """List all states available in the electoral rolls dataset.

    Returns:
        List of state names available in the data.

    Examples:
        >>> states = list_available_states()
        >>> len(states)
        34
        >>> "Delhi" in states
        True
    """
    from ._utils import load_electoral_data

    electoral_data = load_electoral_data()
    # Get column names, excluding non-state columns
    state_cols = [
        str(col)
        for col in electoral_data.columns
        if not str(col).startswith("__") and str(col) != "total_n"
    ]
    return sorted(state_cols)
