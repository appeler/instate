"""National candidate validation rejects missing state probabilities."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_training.prep_er_data.name_tables import V2_STATE_ORDER
from model_training.prep_er_data.validate_national_candidate import validate

if TYPE_CHECKING:
    from pathlib import Path


def sha256(path: Path) -> str:
    """Return a test artifact's digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_candidate(
    root: Path, *, null_state: str | None = None
) -> tuple[Path, Path, Path]:
    """Write a one-surname candidate spanning all 35 states."""
    inputs = root / "inputs"
    inputs.mkdir()
    states = {}
    for index, state in enumerate(V2_STATE_ORDER):
        source = inputs / f"state_{index}.csv.gz"
        with gzip.open(source, "wt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["last_name", "n_times"])
            writer.writerow(["example", 3])
        states[state] = {
            "filename": source.name,
            "record_weight": 3,
            "sha256": sha256(source),
        }
    manifest = {"states": states, "total_input_record_weight": 3 * len(states)}
    (inputs / "input_manifest.json").write_text(json.dumps(manifest))

    training = root / "training.csv.gz"
    with gzip.open(training, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["last_name", "state", "n_times"])
        writer.writerows(("example", state, 3) for state in V2_STATE_ORDER)

    columns: dict[str, list[object]] = {"last_name": ["example"]}
    for state in V2_STATE_ORDER:
        columns[state] = [None if state == null_state else 1 / len(V2_STATE_ORDER)]
    columns["total_n"] = [3 * len(V2_STATE_ORDER)]
    lookup = root / "lookup.parquet"
    pq.write_table(pa.table(columns), lookup)
    return inputs, training, lookup


def test_valid_candidate_reconstructs_every_state(tmp_path: Path) -> None:
    inputs, training, lookup = write_candidate(tmp_path)
    report = validate(inputs, training, lookup, tmp_path / "validation.json")
    assert not any(report["checks"].values())


def test_null_probability_is_rejected(tmp_path: Path) -> None:
    inputs, training, lookup = write_candidate(tmp_path, null_state=V2_STATE_ORDER[0])
    with pytest.raises(ValueError, match="validation failed"):
        validate(inputs, training, lookup, tmp_path / "validation.json")
