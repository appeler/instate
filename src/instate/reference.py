"""Reference lookups keyed by state rather than by name.

These are auxiliary tables, not name-pattern estimates, so they do not
carry the inference contract's metadata columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import GT_KEYS, STATE_LANGUAGE_ALIASES


def lookup_state_official_languages(
    data: pd.DataFrame | list[str] | str,
    state_column: str | None = None,
) -> pd.DataFrame:
    """Look up each state's official languages.

    Args:
        data: DataFrame of inputs, or a state name string or list.
        state_column: Column holding state names for DataFrame input.

    Returns:
        A copy of the input with an ``official_languages`` column; states
        outside the table receive a missing value.

    Raises:
        TypeError: If ``data`` is not a supported input type.
        ValueError: If ``state_column`` is missing or absent for DataFrame
            input.
    """
    if isinstance(data, pd.DataFrame):
        if state_column is None:
            raise ValueError("state_column is required for DataFrame input")
        if state_column not in data.columns:
            raise ValueError(f"state column {state_column!r} does not exist")
        frame = data.copy()
        column = state_column
    elif isinstance(data, str):
        frame = pd.DataFrame({"state": [data]})
        column = "state"
    elif isinstance(data, list):
        frame = pd.DataFrame({"state": data})
        column = "state"
    else:
        raise TypeError("data must be a DataFrame, list, or string")

    path = Path(__file__).parent / "data" / "state_to_languages.parquet"
    table = pd.read_parquet(path)
    value_columns = [name for name in table.columns if name != "state"]
    keys = frame[column].replace(STATE_LANGUAGE_ALIASES)
    values = pd.DataFrame(table.set_index("state").reindex(keys)[value_columns])
    for name in value_columns:
        frame[name] = values[name].to_numpy()
    return frame


def list_supported_states() -> list[str]:
    """List the states covered by the composition functions.

    Returns:
        State names in vocabulary order.
    """
    return list(GT_KEYS)
