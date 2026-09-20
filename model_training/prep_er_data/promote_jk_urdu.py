"""Promote calibrated Urdu fields onto the reconciled J&K source inventory."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def sha256(path):
    """Return a file's SHA-256 digest."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def promote(
    base, overlays, candidates, relationships, word_audit, source_audit, output
):
    """Build and validate one enriched inventory without changing source rows."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    inventory_out = output / "inventory.parquet"
    provenance_out = output / "field_provenance.parquet"
    db = duckdb.connect()
    db.execute("SET memory_limit='3GB'")
    db.execute("SET threads=4")
    db.read_parquet(str(Path(base).resolve())).create_view("base")
    db.read_parquet([str(Path(path).resolve()) for path in overlays]).create_view(
        "overlay_rows"
    )
    db.read_parquet(str(Path(candidates).resolve())).create_view("candidates")
    db.read_parquet(str(Path(relationships).resolve())).create_view("relationships")
    checks = {
        "base_rows": db.execute("SELECT count(*) FROM base").fetchone()[0],
        "base_duplicate_keys": db.execute(
            "SELECT count(*)-count(DISTINCT entry_event_key) FROM base"
        ).fetchone()[0],
        "candidate_rows": db.execute("SELECT count(*) FROM candidates").fetchone()[0],
        "candidate_duplicate_keys": db.execute(
            "SELECT count(*)-count(DISTINCT entry_event_key || ':' || role) "
            "FROM candidates"
        ).fetchone()[0],
        "relationship_rows": db.execute(
            "SELECT count(*) FROM relationships"
        ).fetchone()[0],
        "relationship_duplicate_keys": db.execute(
            "SELECT count(*)-count(DISTINCT entry_event_key) FROM relationships"
        ).fetchone()[0],
        "candidate_keys_absent_from_base": db.execute(
            "SELECT count(*) FROM candidates c ANTI JOIN base b "
            "ON c.entry_event_key=b.entry_event_key AND c.filename=b.filename"
        ).fetchone()[0],
        "relationship_keys_absent_from_base": db.execute(
            "SELECT count(*) FROM relationships r ANTI JOIN base b "
            "ON r.entry_event_key=b.entry_event_key AND r.filename=b.filename"
        ).fetchone()[0],
        "relationship_source_mismatches": db.execute(
            "SELECT count(*) FROM relationships r JOIN base b "
            "ON r.entry_event_key=b.entry_event_key AND r.filename=b.filename "
            "WHERE r.source_sha256<>b.source_sha256 OR r.page<>b.page"
        ).fetchone()[0],
    }
    db.execute(
        "CREATE TEMP TABLE overlay AS SELECT filename,entry_event_key,"
        "first(elector_name ORDER BY elector_name) FILTER "
        "(WHERE elector_name IS NOT NULL) "
        "AS elector_name,"
        "first(name_candidate ORDER BY name_candidate) FILTER "
        "(WHERE name_candidate IS NOT NULL) AS name_candidate,"
        "first(name_issue ORDER BY name_issue) FILTER (WHERE name_issue IS NOT NULL) "
        "AS name_issue,"
        "first(relative_name ORDER BY relative_name) FILTER "
        "(WHERE relative_name IS NOT NULL) AS relative_name,"
        "first(relative_candidate ORDER BY relative_candidate) FILTER "
        "(WHERE relative_candidate IS NOT NULL) AS relative_candidate,"
        "first(relative_issue ORDER BY relative_issue) FILTER "
        "(WHERE relative_issue IS NOT NULL) AS relative_issue,"
        "first(relative_type ORDER BY relative_type) FILTER "
        "(WHERE relative_type IS NOT NULL) AS relative_type,"
        "count(DISTINCT elector_name) FILTER (WHERE elector_name IS NOT NULL) "
        "+ count(DISTINCT relative_name) FILTER (WHERE relative_name IS NOT NULL) "
        "+ count(DISTINCT relative_type) FILTER (WHERE relative_type IS NOT NULL) "
        "AS value_count FROM overlay_rows GROUP BY filename,entry_event_key"
    )
    checks["conflicting_overlay_fields"] = db.execute(
        "SELECT coalesce(sum(value_count>3),0) FROM overlay"
    ).fetchone()[0]
    db.execute(
        "CREATE TEMP TABLE candidate_fields AS SELECT filename,entry_event_key,"
        "max(candidate) FILTER (WHERE role='own') AS own_candidate,"
        "max(candidate) FILTER (WHERE role='relative') AS relative_candidate "
        "FROM candidates GROUP BY filename,entry_event_key"
    )
    checks["candidate_fields_that_would_replace_existing_names"] = db.execute(
        "SELECT count(*) FROM candidate_fields c JOIN overlay o USING "
        "(filename,entry_event_key) WHERE (c.own_candidate IS NOT NULL "
        "AND o.elector_name IS NOT NULL) OR (c.relative_candidate IS NOT NULL "
        "AND o.relative_name IS NOT NULL)"
    ).fetchone()[0]
    if any(
        checks[key]
        for key in (
            "base_duplicate_keys",
            "candidate_duplicate_keys",
            "relationship_duplicate_keys",
            "candidate_keys_absent_from_base",
            "relationship_keys_absent_from_base",
            "relationship_source_mismatches",
            "conflicting_overlay_fields",
            "candidate_fields_that_would_replace_existing_names",
        )
    ):
        raise ValueError(f"Promotion invariant failed: {checks}")
    columns = pq.ParquetFile(base).schema_arrow.names
    replacements = {
        "elector_name": "coalesce(o.elector_name,c.own_candidate,b.elector_name)",
        "name_candidate": "coalesce(o.name_candidate,c.own_candidate,b.name_candidate)",
        "name_issue": (
            "CASE WHEN coalesce(o.elector_name,c.own_candidate) IS NOT NULL "
            "THEN NULL ELSE coalesce(o.name_issue,b.name_issue) END"
        ),
        "relative_name": (
            "coalesce(o.relative_name,c.relative_candidate,b.relative_name)"
        ),
        "relative_candidate": (
            "coalesce(o.relative_candidate,c.relative_candidate,b.relative_candidate)"
        ),
        "relative_issue": (
            "CASE WHEN coalesce(o.relative_name,c.relative_candidate) IS NOT NULL "
            "THEN NULL ELSE coalesce(o.relative_issue,b.relative_issue) END"
        ),
        "relative_type": "coalesce(o.relative_type,r.relationship,b.relative_type)",
    }
    select = ",".join(
        f"{replacements.get(column, f'b.{column}')} AS {column}" for column in columns
    )
    inventory_query = (
        "SELECT "  # noqa: S608 -- identifiers come from the local Parquet schema.
        + select
        + " FROM base b LEFT JOIN overlay o USING (filename,entry_event_key) "
        "LEFT JOIN candidate_fields c USING (filename,entry_event_key) "
        "LEFT JOIN relationships r USING (filename,entry_event_key) "
        "ORDER BY b.filename,b.event_key"
    )
    db.execute(
        f"COPY ({inventory_query}) TO ? "
        "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)",
        [str(inventory_out.resolve())],
    )
    db.execute(
        "COPY (SELECT c.filename,c.entry_event_key,c.role,c.page,c.region,"
        "c.candidate,c.direct_ocr_read,c.direct_ocr_candidate,c.review_status,"
        "CASE WHEN c.role='own' THEN coalesce(o.name_issue,b.name_issue) "
        "ELSE coalesce(o.relative_issue,b.relative_issue) END AS prior_issue,"
        "'muse_calibrated_word_vote' AS recovery_method,"
        "'muse-spark-1.3-contributor' AS model,6 AS minimum_support,"
        "0.75::DOUBLE AS minimum_share,1 AS minimum_margin "
        "FROM candidates c JOIN base b USING (filename,entry_event_key) "
        "LEFT JOIN overlay o USING (filename,entry_event_key) "
        "ORDER BY c.filename,c.entry_event_key,c.role) TO ? "
        "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)",
        [str(provenance_out.resolve())],
    )
    db.read_parquet(str(inventory_out.resolve())).create_view("promoted")
    checks.update(
        {
            "output_rows": pq.ParquetFile(inventory_out).metadata.num_rows,
            "provenance_rows": pq.ParquetFile(provenance_out).metadata.num_rows,
            "output_duplicate_keys": db.execute(
                "SELECT count(*)-count(DISTINCT entry_event_key) FROM promoted"
            ).fetchone()[0],
            "immutable_rows_changed": db.execute(
                "SELECT count(*) FROM (SELECT filename,source_sha256,event_key,"
                "event_type,"
                "supplement,number,number_raw,id,entry_event_key,page,bbox,"
                "assembly_eligible,deleted_stamp,active,change_event_keys FROM base "
                "EXCEPT SELECT filename,source_sha256,event_key,event_type,supplement,"
                "number,number_raw,id,entry_event_key,page,bbox,assembly_eligible,"
                "deleted_stamp,active,change_event_keys FROM promoted)"
            ).fetchone()[0],
            "active_rows": db.execute(
                "SELECT count(*) FROM promoted WHERE active"
            ).fetchone()[0],
            "active_assembly_rows": db.execute(
                "SELECT count(*) FROM promoted WHERE active AND assembly_eligible"
            ).fetchone()[0],
            "accepted_own_names": db.execute(
                "SELECT count(*) FROM promoted WHERE elector_name IS NOT NULL"
            ).fetchone()[0],
            "accepted_relative_names": db.execute(
                "SELECT count(*) FROM promoted WHERE relative_name IS NOT NULL"
            ).fetchone()[0],
            "accepted_relationships": db.execute(
                "SELECT count(*) FROM promoted WHERE relative_type IS NOT NULL"
            ).fetchone()[0],
        }
    )
    db.close()
    if (
        checks["output_rows"] != checks["base_rows"]
        or checks["output_duplicate_keys"]
        or checks["immutable_rows_changed"]
        or checks["provenance_rows"] != checks["candidate_rows"]
    ):
        raise ValueError(f"Output validation failed: {checks}")
    source_report = json.loads(Path(source_audit).read_text())
    summary = dict(source_report["summary"])
    summary["accepted_urdu_names"] = checks["accepted_own_names"]
    parts = []
    for source_part in source_report["parts"]:
        if not (source_part.get("counts", {}).get("inventory") or 0):
            continue
        part = dict(source_part)
        part.update(source_part.get("counts", {}))
        parts.append(part)
    checks["source_parts_without_inventory"] = len(source_report["parts"]) - len(parts)
    scripts = dict(source_report["script_sha256"])
    scripts[Path(__file__).name] = sha256(Path(__file__))
    audit = {
        "status": "calibrated_urdu_recovery_promoted",
        "summary": summary,
        "parts": parts,
        "script_sha256": scripts,
        "checks": checks,
        "vote_policy": {
            "minimum_distinct_records_and_calls": 6,
            "minimum_share": 0.75,
            "minimum_margin": 1,
        },
        "word_audit": json.loads(Path(word_audit).read_text())["control_test"],
        "inputs": {
            "base_inventory": sha256(base),
            "overlay_inventories": {str(path): sha256(path) for path in overlays},
            "candidates": sha256(candidates),
            "relationships": sha256(relationships),
            "word_audit": sha256(word_audit),
            "source_audit": sha256(source_audit),
        },
        "artifacts": {
            "inventory.parquet": sha256(inventory_out),
            "field_provenance.parquet": sha256(provenance_out),
        },
        "limitations": [
            "Hidden controls estimate word-reading accuracy, not surname accuracy.",
            "Unsupported geometry and evidence below the calibrated gate remain null.",
            "Cross-script Hindi and Urdu editions require identity deduplication.",
        ],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    return audit


def main():
    """Promote calibrated Urdu recovery fields from command-line inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--relationships", type=Path, required=True)
    parser.add_argument("--word-audit", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(
        args.base,
        args.overlay,
        args.candidates,
        args.relationships,
        args.word_audit,
        args.source_audit,
        args.output,
    )
    sys.stdout.write(
        json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
