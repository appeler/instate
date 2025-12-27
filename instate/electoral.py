"""
Electoral rolls based name-to-state lookup.

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
            If None and DataFrame has 'name' or 'lastname', uses that.

    Returns:
        DataFrame with original data plus 31 state probability columns.
        State columns are named by state (e.g., 'delhi', 'punjab').
        Values are proportions (0-1) representing P(state|lastname).

    Examples:
        >>> names = ["dhingra", "sood", "gowda"]
        >>> result = get_state_distribution(names)
        >>> result[["name", "delhi", "punjab", "karnataka"]]

        >>> df = pd.DataFrame({"lastname": ["dhingra", "sood"]})
        >>> result = get_state_distribution(df, "lastname")
        >>> result.columns[:5].tolist()
    """
    from ._utils import clean_names_in_df, load_electoral_data, prepare_name_dataframe

    # Convert to DataFrame if needed
    df = prepare_name_dataframe(names, name_column)

    # Clean names for matching
    df = clean_names_in_df(df, df.columns[0])

    # Load electoral rolls data
    electoral_data = load_electoral_data()

    # Merge to get state distributions
    # Electoral data has __last_name as key
    result = pd.merge(
        df, electoral_data, left_on="__cleaned_name", right_on="__last_name", how="left"
    )

    # Drop temporary columns
    result = result.drop(columns=["__cleaned_name", "__last_name"], errors="ignore")

    return result


def get_state_languages(
    states: pd.DataFrame | list[str], state_column: str | None = None
) -> pd.DataFrame:
    """Map Indian states to their official languages.

    Based on census data, returns the official language(s) for each state.

    Args:
        states: DataFrame containing states or list of state names.
        state_column: If states is a DataFrame, the column containing state names.

    Returns:
        DataFrame with state and official_languages columns.
        If input was DataFrame, adds official_languages column.

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
            # Try to find state column
            possible_cols = [str(c) for c in df.columns if "state" in str(c).lower()]
            if not possible_cols:
                raise ValueError("state_column must be specified for DataFrame input")
            state_col = possible_cols[0]
        else:
            state_col = state_column

    # Load state-language mapping
    state_lang_path = Path(__file__).parent / "data" / "state_to_languages.csv"
    state_lang_map = pd.read_csv(str(state_lang_path))  # type: ignore[misc]

    # Merge to add languages
    result = df.merge(state_lang_map, left_on=state_col, right_on="state", how="left")

    # Clean up duplicate state column if needed
    if state_col != "state" and "state_y" not in result.columns:
        result = result.drop(columns=["state"], errors="ignore")

    return result


def list_available_states() -> list[str]:
    """List all states available in the electoral rolls dataset.

    Returns:
        List of state names available in the data.

    Examples:
        >>> states = list_available_states()
        >>> len(states)
        31
        >>> "Delhi" in states
        True
    """
    from ._utils import load_electoral_data

    electoral_data = load_electoral_data()
    # Get column names, excluding non-state columns
    state_cols = [
        str(col)
        for col in electoral_data.columns
        if not str(col).startswith("__") and str(col) != "last_name"
    ]
    return sorted(state_cols)
