"""Reference lookups keyed by state."""

from __future__ import annotations

import pandas as pd
import pytest

from instate import list_supported_states, lookup_state_official_languages
from instate.constants import GT_KEYS


def test_list_supported_states_matches_vocabulary():
    states = list_supported_states()
    assert states == list(GT_KEYS)
    assert "Telangana" in states


def test_official_languages_for_known_states():
    result = lookup_state_official_languages(["Delhi", "Punjab"])
    assert result.official_languages.tolist() == ["Hindi, English", "Punjabi"]


def test_official_languages_dataframe_input():
    data = pd.DataFrame({"st": ["Kerala"], "keep": [7]})
    result = lookup_state_official_languages(data, "st")
    assert result.keep.tolist() == [7]
    assert result.official_languages.iloc[0] == "Malayalam"
    assert "official_languages" not in data.columns


def test_unknown_state_gets_missing_value():
    result = lookup_state_official_languages(["Atlantis"])
    assert pd.isna(result.official_languages.iloc[0])


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="state_column is required"):
        lookup_state_official_languages(pd.DataFrame({"state": ["Goa"]}))
    with pytest.raises(ValueError, match="does not exist"):
        lookup_state_official_languages(pd.DataFrame({"a": [1]}), "state")
    with pytest.raises(TypeError, match="must be a DataFrame"):
        lookup_state_official_languages(3.14)  # type: ignore[arg-type]
