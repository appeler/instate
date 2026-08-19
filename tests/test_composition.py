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
    except Exception:  # any resolution failure means skip
        return False
    return True


needs_model = pytest.mark.skipif(
    not model_available(), reason="model artifacts not resolvable"
)


class TestLookupStateComposition:
    def test_known_surname_scores_and_sums_to_one(self):
        result = lookup_state_composition([KNOWN])
        row = result.iloc[0]
        assert bool(row.scored)
        assert not bool(row.abstained)
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


class TestArtifactIntegrity:
    @pytest.fixture(autouse=True)
    def isolated_cache(self, monkeypatch):
        from instate import composition

        monkeypatch.setattr(composition, "_CACHE", {})

    def test_language_shares_hash_mismatch_is_fatal(self, monkeypatch):
        from instate import composition

        monkeypatch.setattr(composition, "_sha256", lambda path: "not-the-hash")
        with pytest.raises(RuntimeError, match="pinned SHA-256"):
            composition._language_shares()

    def test_electoral_table_schema_is_checked(self, monkeypatch):
        from instate import composition

        monkeypatch.setattr(
            pd, "read_parquet", lambda path: pd.DataFrame({"last_name": []})
        )
        with pytest.raises(RuntimeError, match="state vocabulary"):
            composition._electoral_table()

    def test_language_share_state_coverage_is_checked(self, monkeypatch):
        from instate import composition

        stub = pd.DataFrame(
            {"state": ["Atlantis"], "language": ["hindi"], "share": [1.0]}
        )
        monkeypatch.setattr(pd, "read_parquet", lambda path: stub)
        with pytest.raises(RuntimeError, match="cover the state vocabulary"):
            composition._language_shares()

    def test_nonpositive_temperature_is_fatal(self, monkeypatch, tmp_path):
        import json

        from instate import _resources, composition

        bad = tmp_path / "instate_state_lstm_calibration.json"
        bad.write_text(json.dumps({"temperature": 0}), encoding="utf-8")
        monkeypatch.setattr(_resources, "resolve_model", lambda name: str(bad))
        with pytest.raises(RuntimeError, match="temperature must be positive"):
            composition._calibrated_model()


class TestLanguageModelBasisEdges:
    def test_model_basis_short_name_abstains_without_model(self):
        result = estimate_language_composition(["ab"], basis="model")
        row = result.iloc[0]
        assert not bool(row.scored)
        assert row.abstention_reason == "insufficient-evidence"


class TestReviewFindings:
    """Regression tests for the 3.0.0 independent-review findings."""

    def test_series_input_preserves_custom_index(self):
        result = lookup_state_composition(pd.Series([KNOWN], index=[100]))
        assert list(result.index) == [100]

    def test_auto_basis_prefers_lookup_for_short_in_table_names(self, monkeypatch):
        from instate import composition

        table = composition._electoral_table()
        short = table.iloc[[0]].copy()
        short.index = ["om"]
        monkeypatch.setitem(composition._CACHE, "electoral", pd.concat([table, short]))
        result = estimate_language_composition(["om"], basis="auto")
        row = result.iloc[0]
        assert bool(row.scored)
        assert row.language_basis == "electoral-lookup"

    def test_language_shares_pin_defeats_consistent_tampering(
        self, monkeypatch, tmp_path
    ):
        import hashlib
        import json
        import shutil

        from instate import composition

        source = composition._DATA_DIRECTORY
        for name in (
            "state_language_shares.parquet",
            "state_language_shares.manifest.json",
        ):
            shutil.copy(source / name, tmp_path / name)
        parquet = tmp_path / "state_language_shares.parquet"
        parquet.write_bytes(parquet.read_bytes() + b"tampered")
        manifest_path = tmp_path / "state_language_shares.manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["artifact"]["sha256"] = hashlib.sha256(
            parquet.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        monkeypatch.setattr(composition, "_DATA_DIRECTORY", tmp_path)
        monkeypatch.setattr(composition, "_CACHE", {})
        with pytest.raises(RuntimeError, match="pinned"):
            composition._language_shares()
