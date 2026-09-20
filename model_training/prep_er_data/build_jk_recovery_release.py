"""Build final national inputs from English, Hindi, and calibrated Urdu handoffs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import duckdb

from model_training.prep_er_data.name_tables import FILE2STATE, V2_STATE_ORDER

HINDI_PREFIX = "jk-hindi-2018:"
URDU_PREFIX = "jk-urdu-2018-calibrated:"


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_hash(audit: Path, artifact: Path) -> None:
    """Require an artifact to match the audit beside its handoff."""
    report = json.loads(audit.read_text())
    expected = report["artifacts"][artifact.name]
    if sha256(artifact) != expected:
        raise ValueError(f"Artifact hash mismatch: {artifact}")


def write_counts(path: Path, counts: list[tuple[str, int]]) -> None:
    """Write one deterministic state surname-count table."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["last_name", "n_times"])
    writer.writerows(counts)
    path.write_bytes(
        gzip.compress(buffer.getvalue().encode(), compresslevel=9, mtime=0)
    )


def read_total(path: Path) -> tuple[int, int]:
    """Return unique strings and total weight from one state table."""
    with gzip.open(path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(set(row) != {"last_name", "n_times"} for row in rows):
        raise ValueError(f"Unexpected state-table columns: {path}")
    if len({row["last_name"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate surname in state table: {path}")
    weights = [int(row["n_times"]) for row in rows]
    if any(weight < 1 for weight in weights):
        raise ValueError(f"Nonpositive state-table weight: {path}")
    return len(rows), sum(weights)


def validate_link_contract(
    connection: duckdb.DuckDBPyConnection, expected_links: int
) -> None:
    """Require unique link keys that each join once to both handoffs."""
    integrity = connection.execute(
        "SELECT count(*)-count(DISTINCT hindi_entry_event_key),"
        "count(*)-count(DISTINCT urdu_entry_event_key),"
        "count(*) FILTER (WHERE hindi_entry_event_key IS NULL "
        "OR urdu_entry_event_key IS NULL) FROM links"
    ).fetchone()
    if any(integrity):
        raise ValueError(
            f"Cross-script links are not one-to-one and complete: {integrity}"
        )
    joined = connection.execute(
        "SELECT count(*) FROM links l "
        "JOIN hindi h ON h.entry_event_key=l.hindi_entry_event_key "
        "JOIN urdu u ON u.entry_event_key=l.urdu_entry_event_key"
    ).fetchone()[0]
    if joined != expected_links:
        raise ValueError("Cross-script links do not join exactly once to both handoffs")


def build_inputs(
    base_inputs: Path,
    output: Path,
    english: Path,
    english_audit: Path,
    hindi: Path,
    hindi_audit: Path,
    urdu: Path,
    urdu_audit: Path,
    links: Path,
    links_audit: Path,
) -> dict[str, object]:
    """Copy current state inputs and replace JK with a linked-edition union."""
    if output.exists():
        raise FileExistsError(output)
    expected_files = {f"last_names_{slug}.csv.gz" for slug in FILE2STATE}
    actual_files = {path.name for path in base_inputs.glob("last_names_*.csv.gz")}
    states_match = set(FILE2STATE.values()) == set(V2_STATE_ORDER)
    if actual_files != expected_files or not states_match:
        raise ValueError("Base state inputs do not contain the exact 35-state contract")
    artifact_hash(english_audit, english)
    artifact_hash(hindi_audit, hindi)
    artifact_hash(urdu_audit, urdu)
    links_report = json.loads(links_audit.read_text())
    link_record = links_report["edition_links"]
    if sha256(links) != link_record["sha256"] or not link_record["one_to_one"]:
        raise ValueError("Cross-script edition links do not match their audit")
    output.mkdir()
    for path in base_inputs.glob("last_names_*.csv.gz"):
        shutil.copy2(path, output / path.name)
    with duckdb.connect() as connection:
        connection.read_parquet(str(english)).create_view("english")
        connection.read_parquet(str(hindi)).create_view("hindi_raw")
        connection.read_parquet(str(urdu)).create_view("urdu_raw")
        connection.read_parquet(str(links)).create_view("links")
        invalid_prefixes = connection.execute(
            "SELECT "
            "(SELECT count(*) FROM hindi_raw WHERE "
            "NOT starts_with(elector_id,?)),"
            "(SELECT count(*) FROM urdu_raw WHERE "
            "NOT starts_with(elector_id,?))",
            [HINDI_PREFIX, URDU_PREFIX],
        ).fetchone()
        if any(invalid_prefixes):
            raise ValueError(f"Unexpected J&K elector ID prefix: {invalid_prefixes}")
        connection.execute(
            "CREATE VIEW hindi AS SELECT substring(elector_id,15) "
            "AS entry_event_key,* FROM hindi_raw"
        )
        connection.execute(
            "CREATE VIEW urdu AS SELECT substring(elector_id,25) "
            "AS entry_event_key,* FROM urdu_raw"
        )
        source_counts = connection.execute(
            "SELECT (SELECT count(*) FROM english),"
            "(SELECT count(*) FROM hindi),(SELECT count(*) FROM urdu),"
            "(SELECT count(*) FROM links),"
            "(SELECT count(*) FROM english WHERE surname_latin_normalized IS NOT NULL),"
            "(SELECT count(*) FROM hindi WHERE surname_latin_normalized IS NOT NULL),"
            "(SELECT count(*) FROM urdu WHERE surname_latin_normalized IS NOT NULL)"
        ).fetchone()
        if source_counts[:4] != (169_191, 2_230_975, 4_608_102, 693_201):
            raise ValueError(f"Unexpected handoff or link rows: {source_counts[:4]}")
        validate_link_contract(connection, source_counts[3])
        connection.execute(
            "CREATE TEMP TABLE linked AS SELECT l.*,"
            "h.surname_latin_normalized AS hindi_name,"
            "u.surname_latin_normalized AS urdu_name "
            "FROM links l JOIN hindi h ON h.entry_event_key=l.hindi_entry_event_key "
            "JOIN urdu u ON u.entry_event_key=l.urdu_entry_event_key"
        )
        link_counts = connection.execute(
            "SELECT count(*),count(*) FILTER (WHERE hindi_name IS NOT NULL),"
            "count(*) FILTER (WHERE urdu_name IS NOT NULL),"
            "count(*) FILTER (WHERE hindi_name IS NOT NULL AND urdu_name IS NOT NULL),"
            "count(*) FILTER (WHERE hindi_name IS NULL AND urdu_name IS NOT NULL),"
            "count(*) FILTER (WHERE hindi_name IS NOT NULL AND urdu_name IS NOT NULL "
            "AND hindi_name<>urdu_name) FROM linked"
        ).fetchone()
        connection.execute(
            "CREATE TEMP TABLE chosen AS "
            "SELECT surname_latin_normalized AS last_name,'english' AS source "
            "FROM english WHERE surname_latin_normalized IS NOT NULL UNION ALL "
            "SELECT surname_latin_normalized,'hindi' FROM hindi "
            "WHERE surname_latin_normalized IS NOT NULL UNION ALL "
            "SELECT u.surname_latin_normalized,'urdu' FROM urdu u "
            "LEFT JOIN links l ON u.entry_event_key=l.urdu_entry_event_key "
            "LEFT JOIN hindi h ON h.entry_event_key=l.hindi_entry_event_key "
            "WHERE u.surname_latin_normalized IS NOT NULL "
            "AND (l.urdu_entry_event_key IS NULL "
            "OR h.surname_latin_normalized IS NULL)"
        )
        invalid = connection.execute(
            "SELECT count(*) FROM chosen WHERE NOT "
            "regexp_full_match(last_name,'[a-z]+')"
        ).fetchone()[0]
        if invalid:
            raise ValueError("Chosen JK rows contain non-ASCII normalized surnames")
        by_source = dict(
            sorted(
                connection.execute(
                    "SELECT source,count(*) FROM chosen GROUP BY source"
                ).fetchall()
            )
        )
        counts = connection.execute(
            "SELECT last_name,count(*)::BIGINT AS n FROM chosen "
            "GROUP BY last_name ORDER BY last_name"
        ).fetchall()
    write_counts(output / "last_names_jk.csv.gz", counts)
    input_summary = {}
    for slug, state in FILE2STATE.items():
        path = output / f"last_names_{slug}.csv.gz"
        strings, weight = read_total(path)
        input_summary[state] = {
            "filename": path.name,
            "strings": strings,
            "record_weight": weight,
            "sha256": sha256(path),
        }
    if input_summary[FILE2STATE["jk"]]["record_weight"] != sum(by_source.values()):
        raise ValueError("JK aggregate does not reconstruct its selected sources")
    manifest = {
        "status": "final_recovery_inputs",
        "policy": (
            "Count each audited Hindi/Urdu edition link once. Prefer a mapped Hindi "
            "selection when both scripts select; otherwise retain the mapped Urdu "
            "selection. Preserve all unlinked records and the disjoint English handoff."
        ),
        "handoffs": {
            "english": {
                "path": str(english),
                "sha256": sha256(english),
                "rows": source_counts[0],
                "mapped_selections": source_counts[4],
            },
            "hindi": {
                "path": str(hindi),
                "sha256": sha256(hindi),
                "rows": source_counts[1],
                "mapped_selections": source_counts[5],
            },
            "urdu": {
                "path": str(urdu),
                "sha256": sha256(urdu),
                "rows": source_counts[2],
                "mapped_selections": source_counts[6],
            },
        },
        "edition_links": {
            "path": str(links),
            "sha256": sha256(links),
            "rows": link_counts[0],
            "hindi_mapped": link_counts[1],
            "urdu_mapped": link_counts[2],
            "both_mapped": link_counts[3],
            "urdu_fills_unmapped_hindi": link_counts[4],
            "different_mapped_spellings": link_counts[5],
        },
        "jk_selected_by_source": by_source,
        "jk_unique_latin_strings": len(counts),
        "jk_selected_record_weight": sum(by_source.values()),
        "states": input_summary,
        "total_input_record_weight": sum(
            state["record_weight"] for state in input_summary.values()
        ),
        "producer_sha256": sha256(Path(__file__)),
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Build the final 35-state input directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("english", "hindi", "urdu"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-audit", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--links-audit", type=Path, required=True)
    args = parser.parse_args()
    result = build_inputs(
        args.base_inputs,
        args.output,
        args.english,
        args.english_audit,
        args.hindi,
        args.hindi_audit,
        args.urdu,
        args.urdu_audit,
        args.links,
        args.links_audit,
    )
    sys.stdout.write(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "jk_selected_by_source",
                    "jk_unique_latin_strings",
                    "jk_selected_record_weight",
                    "total_input_record_weight",
                )
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
