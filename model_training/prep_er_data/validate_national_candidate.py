"""Validate that a national lookup exactly reconstructs its retained state counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from model_training.prep_er_data.name_tables import V2_STATE_ORDER


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(
    inputs: Path, training: Path, lookup: Path, output: Path
) -> dict[str, object]:
    """Check source hashes, retained cells, totals, shares, and output hashes."""
    manifest_path = inputs / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for state, record in manifest["states"].items():
        path = inputs / record["filename"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Input hash mismatch: {state}")
    expected_columns = ["last_name", *V2_STATE_ORDER, "total_n"]
    if pq.read_schema(lookup).names != expected_columns:
        raise ValueError("Lookup columns do not match the 35-state contract")
    with duckdb.connect() as connection:
        connection.read_csv(str(training), header=True).create_view("training")
        connection.read_parquet(str(lookup)).create_view("lookup")
        training_states = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT state FROM training"
            ).fetchall()
        }
        checks = {
            "training_states_missing_or_extra": len(
                training_states.symmetric_difference(V2_STATE_ORDER)
            ),
            "duplicate_training_cells": connection.execute(
                "SELECT count(*)-count(DISTINCT (last_name,state)) FROM training"
            ).fetchone()[0],
            "invalid_training_rows": connection.execute(
                "SELECT count(*) FROM training WHERE n_times<3 OR "
                "NOT regexp_full_match(last_name,'[a-z]+') OR length(last_name)<=2"
            ).fetchone()[0],
            "duplicate_lookup_names": connection.execute(
                "SELECT count(*)-count(DISTINCT last_name) FROM lookup"
            ).fetchone()[0],
            "lookup_names_without_training": connection.execute(
                "SELECT count(*) FROM lookup l ANTI JOIN "
                "(SELECT DISTINCT last_name FROM training) t USING(last_name)"
            ).fetchone()[0],
            "training_names_without_lookup": connection.execute(
                "SELECT count(*) FROM (SELECT DISTINCT last_name FROM training) t "
                "ANTI JOIN lookup l USING(last_name)"
            ).fetchone()[0],
            "lookup_total_mismatches": connection.execute(
                "SELECT count(*) FROM lookup l JOIN "
                "(SELECT last_name,sum(n_times) n FROM training GROUP BY last_name) t "
                "USING(last_name) WHERE l.total_n<>t.n"
            ).fetchone()[0],
        }
        probability_sum = "+".join(f'"{state}"' for state in V2_STATE_ORDER)
        null_probability = " OR ".join(f'"{state}" IS NULL' for state in V2_STATE_ORDER)
        checks["invalid_probability_rows"] = connection.execute(
            f"SELECT count(*) FROM lookup WHERE "  # noqa: S608
            f"({null_probability}) OR abs(({probability_sum})-1)>1e-9"
        ).fetchone()[0]
        per_state = {}
        cell_mismatches = 0
        invalid_probabilities = 0
        for state in V2_STATE_ORDER:
            training_total = connection.execute(
                "SELECT coalesce(sum(n_times),0) FROM training WHERE state=?", [state]
            ).fetchone()[0]
            query = (
                f'SELECT coalesce(sum(round("{state}"*total_n)),0),'  # noqa: S608
                f'count(*) FILTER (WHERE "{state}" IS NULL '
                f'OR "{state}"<0 OR "{state}">1 '
                f'OR NOT isfinite("{state}")) FROM lookup'
            )
            reconstructed, invalid = connection.execute(query).fetchone()
            mismatch_query = (
                f"SELECT count(*) FROM lookup l LEFT JOIN training t "  # noqa: S608
                f"ON l.last_name=t.last_name AND t.state=? WHERE "
                f'"{state}" IS NULL OR '
                f'abs("{state}"*total_n-coalesce(t.n_times,0))>1e-6'
            )
            mismatches = connection.execute(mismatch_query, [state]).fetchone()[0]
            input_record = manifest["states"][state]
            per_state[state] = {
                "input_record_weight": input_record["record_weight"],
                "retained_training_weight": training_total,
                "reconstructed_lookup_weight": int(reconstructed),
                "excluded_by_national_filters": (
                    input_record["record_weight"] - training_total
                ),
                "cell_mismatches": mismatches,
            }
            cell_mismatches += mismatches
            invalid_probabilities += invalid
        checks["state_cell_mismatches"] = cell_mismatches
        checks["invalid_probability_values"] = invalid_probabilities
        checks["state_total_mismatches"] = sum(
            row["retained_training_weight"] != row["reconstructed_lookup_weight"]
            for row in per_state.values()
        )
        counts = {
            "lookup_surnames": pq.ParquetFile(lookup).metadata.num_rows,
            "training_cells": connection.execute(
                "SELECT count(*) FROM training"
            ).fetchone()[0],
            "retained_record_weight": connection.execute(
                "SELECT sum(n_times) FROM training"
            ).fetchone()[0],
            "input_record_weight": manifest["total_input_record_weight"],
        }
    if any(checks.values()) or any(
        not math.isfinite(value)
        for row in per_state.values()
        for value in row.values()
        if isinstance(value, float)
    ):
        raise ValueError(f"National candidate validation failed: {checks}")
    report = {
        "status": "validated_final_recovery_candidate",
        "checks": checks,
        "counts": counts,
        "per_state": per_state,
        "inputs": {
            "manifest": sha256(manifest_path),
            "training.csv.gz": sha256(training),
            "lookup.parquet": sha256(lookup),
        },
        "producer_sha256": sha256(Path(__file__)),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Validate one candidate from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.inputs, args.training, args.lookup, args.output)
    summary = {"status": report["status"], **report["counts"]}
    sys.stdout.write(json.dumps(summary, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
