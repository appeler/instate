"""Composition results: lookup, model estimates, and language mixing."""

from __future__ import annotations

import pandas as pd
import pytest

from instate import (
    estimate_language_composition,
    estimate_state_composition,
    lookup_state_composition,
)
from instate.composition import STATE_SHARE_COLUMNS
from instate.constants import GT_KEYS

KNOWN = "sood"
UNKNOWN = "xqzzyqqw"

STATE_COLUMNS = list(STATE_SHARE_COLUMNS.values())

CONTRACT_COLUMNS = [
    "inference_contract_version",
    "estimate_type",
    "result_form",
    "target",
    "input_scope",
    "scored",
    "script_supported",
    "abstained",
    "abstention_reason",
    "model_id",
    "model_version",
    "model_revision",
    "reference_population",
    "calibration_status",
    "calibration_reference",
    "uncertainty_method",
    "uncertainty_level",
]


def model_available() -> bool:
    """True when the pinned model and calibration artifacts resolve."""
    from instate._resources import resolve_model

    try:
        resolve_model("instate_state_lstm.pt")
        resolve_model("instate_state_lstm_calibration.json")
    except Exception:  # noqa: BLE001 - any resolution failure means skip
        return False
    return True


needs_model = pytest.mark.skipif(
    not model_available(), reason="model artifacts not resolvable"
)


class TestLookupStateComposition:
    def test_known_surname_scores_and_sums_to_one(self):
        result = lookup_state_composition([KNOWN])
        row = result.iloc[0]
        assert bool(row.scored) and not bool(row.abstained)
        assert row[STATE_COLUMNS].astype(float).sum() == pytest.approx(1.0)
        assert row.surname_record_count > 0
        assert row.result_form == "composition"
        assert row.inference_contract_version == "1.1"

    def test_contract_columns_present_in_order(self):
        result = lookup_state_composition([KNOWN])
        assert [c for c in result.columns if c in CONTRACT_COLUMNS] == CONTRACT_COLUMNS

    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            (None, "missing-name"),
            ("", "missing-name"),
            ("   ", "missing-name"),
            ("12 34", "no-letters"),
            ("गुप्ता", "unsupported-script"),
            (UNKNOWN, "out-of-dictionary"),
        ],
    )
    def test_abstention_reasons(self, value, reason):
        result = lookup_state_composition([value])
        row = result.iloc[0]
        assert not bool(row.scored)
        assert bool(row.abstained)
        assert row.abstention_reason == reason
        assert pd.isna(row[STATE_COLUMNS]).all()
        assert pd.isna(row.surname_record_count)

    def test_unsupported_script_is_flagged(self):
        result = lookup_state_composition(["गुप्ता"])
        flag = result.script_supported.iloc[0]
        assert not pd.isna(flag)
        assert not bool(flag)

    def test_dataframe_input_preserves_rows_order_and_index(self):
        data = pd.DataFrame(
            {"surname": [KNOWN, UNKNOWN, None], "keep": [1, 2, 3]},
            index=[10, 20, 30],
        )
        result = lookup_state_composition(data, "surname")
        assert list(result.index) == [10, 20, 30]
        assert result["keep"].tolist() == [1, 2, 3]
        assert data.columns.tolist() == ["surname", "keep"]
        assert "scored" not in data.columns

    def test_reserved_input_column_is_preserved(self):
        data = pd.DataFrame({"surname": [KNOWN], "scored": ["mine"]})
        result = lookup_state_composition(data, "surname")
        assert result["input_scored"].tolist() == ["mine"]
        assert bool(result["scored"].iloc[0])

    def test_string_series_and_list_agree(self):
        by_list = lookup_state_composition([KNOWN])
        by_str = lookup_state_composition(KNOWN)
        by_series = lookup_state_composition(pd.Series([KNOWN]))
        for frame in (by_str, by_series):
            assert frame[STATE_COLUMNS].iloc[0].tolist() == pytest.approx(
                by_list[STATE_COLUMNS].iloc[0].tolist()
            )

    def test_case_and_whitespace_normalized(self):
        upper = lookup_state_composition(["  SOOD  "])
        lower = lookup_state_composition([KNOWN])
        assert upper[STATE_COLUMNS].iloc[0].tolist() == pytest.approx(
            lower[STATE_COLUMNS].iloc[0].tolist()
        )

    def test_missing_column_arguments_raise(self):
        with pytest.raises(ValueError, match="surname_column is required"):
            lookup_state_composition(pd.DataFrame({"surname": [KNOWN]}))
        with pytest.raises(ValueError, match="exactly once"):
            lookup_state_composition(pd.DataFrame({"a": [1]}), "missing")
        with pytest.raises(TypeError, match="must be a DataFrame"):
            lookup_state_composition(42)  # type: ignore[arg-type]


class TestLanguageComposition:
    def test_lookup_basis_scores_and_sums_to_one(self):
        result = estimate_language_composition([KNOWN], basis="lookup")
        row = result.iloc[0]
        language_columns = [
            c for c in result.columns if c.startswith("language_share_")
        ]
        assert bool(row.scored)
        assert row.language_basis == "electoral-lookup"
        assert row[language_columns].astype(float).sum() == pytest.approx(1.0)
        assert row.target == "language-composition"

    def test_lookup_basis_abstains_out_of_dictionary(self):
        result = estimate_language_composition([UNKNOWN], basis="lookup")
        row = result.iloc[0]
        assert not bool(row.scored)
        assert row.abstention_reason == "out-of-dictionary"
        assert pd.isna(row.language_basis)

    def test_telugu_surname_leans_telugu(self):
        result = estimate_language_composition(["chintalapati"], basis="lookup")
        assert float(result.language_share_telugu.iloc[0]) > 0.5

    def test_invalid_basis_rejected(self):
        with pytest.raises(ValueError, match="basis"):
            estimate_language_composition([KNOWN], basis="knn")  # type: ignore[arg-type]

    @needs_model
    def test_auto_basis_falls_back_to_model(self):
        result = estimate_language_composition([KNOWN, UNKNOWN], basis="auto")
        assert result.language_basis.tolist() == ["electoral-lookup", "state-model"]
        assert result.scored.all()

    @needs_model
    def test_model_basis_scores_known_names_too(self):
        result = estimate_language_composition([KNOWN], basis="model")
        assert result.language_basis.iloc[0] == "state-model"


@needs_model
class TestEstimateStateComposition:
    def test_out_of_dictionary_surname_is_scored(self):
        result = estimate_state_composition([UNKNOWN])
        row = result.iloc[0]
        assert bool(row.scored)
        assert row[STATE_COLUMNS].astype(float).sum() == pytest.approx(1.0)
        assert row.calibration_status == "temperature-scaled"

    def test_short_input_abstains(self):
        result = estimate_state_composition(["ab"])
        row = result.iloc[0]
        assert not bool(row.scored)
        assert row.abstention_reason == "insufficient-evidence"

    def test_state_columns_cover_vocabulary(self):
        result = estimate_state_composition([KNOWN])
        assert len(STATE_COLUMNS) == len(GT_KEYS)
        assert all(column in result.columns for column in STATE_COLUMNS)
