"""Invariants of the inference-contract metadata layer."""

from __future__ import annotations

import pandas as pd
import pytest

from instate._contract import (
    ResultProvenance,
    metadata_columns,
    preserve_reserved_columns,
)

PROVENANCE = ResultProvenance(
    target="state-composition",
    input_scope="last-name",
    model_id="test",
    model_version="0",
    model_revision="sha256:0",
    reference_population="test rolls",
    calibration_status="not-applicable",
    calibration_reference="not-applicable",
)


def test_columns_are_complete_and_typed():
    columns = metadata_columns(
        PROVENANCE,
        scored=[True, False],
        script_supported=[True, None],
        abstention_reasons=[None, "missing-name"],
    )
    assert columns["inference_contract_version"][0] == "1.1"
    assert columns["result_form"][0] == "composition"
    assert list(columns["abstained"]) == [False, True]
    assert columns["scored"].dtype.name == "boolean"
    assert columns["abstention_reason"].dtype.name == "string"
    assert pd.isna(columns["abstention_reason"][0])
    assert pd.isna(columns["script_supported"][1])


def test_unscored_row_must_abstain():
    with pytest.raises(ValueError, match="unscored row must abstain"):
        metadata_columns(
            PROVENANCE,
            scored=[False],
            script_supported=[True],
            abstention_reasons=[None],
        )


def test_unsupported_script_row_cannot_be_scored():
    with pytest.raises(ValueError, match="cannot be scored"):
        metadata_columns(
            PROVENANCE,
            scored=[True],
            script_supported=[False],
            abstention_reasons=[None],
        )


def test_reason_outside_vocabulary_rejected():
    with pytest.raises(ValueError, match="unknown abstention reason"):
        metadata_columns(
            PROVENANCE,
            scored=[False],
            script_supported=[True],
            abstention_reasons=["mystery"],
        )


def test_misaligned_vectors_rejected():
    with pytest.raises(ValueError, match="must align"):
        metadata_columns(
            PROVENANCE,
            scored=[True, True],
            script_supported=[True],
            abstention_reasons=[None],
        )


def test_reserved_input_columns_are_preserved():
    data = pd.DataFrame(
        {"name": ["sood"], "scored": [1], "input_scored": [2]}
    )
    result = preserve_reserved_columns(data, ["scored"])
    assert list(result.columns) == ["name", "input_scored_1", "input_scored"]
    assert data.columns.tolist() == ["name", "scored", "input_scored"]
