"""Reconcile Urdu card appearances using verified supplement heading outlines.

Names remain unavailable. Source-control differences are reported without changing
records to force a match. Missing required sections withhold the part's inventory.
"""

import argparse
import gzip
import hashlib
import itertools
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ijson

from model_training.prep_er_data.jk_hindi_rows import output_schema, reconcile
from model_training.prep_er_data.jk_urdu_sections import HeaderFonts, classify_pages


def emit(record):
    """Write one machine-readable progress record."""
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


def file_sha256(path):
    """Hash large artifacts with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def open_audit(path):
    """Read plain or losslessly compressed structural evidence."""
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def structural_payloads(audit_path, records):
    """Join ordered source parts and contiguous appearances one PDF at a time."""
    groups = itertools.groupby(records, key=lambda row: row["filename"])
    group = next(groups, None)
    seen = set()
    with open_audit(audit_path) as stream:
        for part in ijson.items(stream, "parts.item", use_float=True):
            name = part["filename"]
            if name in seen:
                raise ValueError("Repeated source part")
            seen.add(name)
            rows = []
            if group is not None and group[0] == name:
                rows = list(group[1])
                group = next(groups, None)
            if len(rows) != (part.get("appearances") or 0):
                raise ValueError(f"Structural row count or order mismatch: {name}")
            yield rows, part
    if group is not None:
        raise ValueError("Structural rows are unordered or absent from the audit")


def select_parts(payloads, filenames):
    """Keep a requested subset while validating every source group."""
    missing = set(filenames) if filenames is not None else None
    for rows, part in payloads:
        if missing is None or part["filename"] in missing:
            if missing is not None:
                missing.remove(part["filename"])
            yield rows, part
    if missing:
        raise ValueError(
            f"Requested parts absent from structural audit: {sorted(missing)}"
        )


def unavailable_part(part, status, issues):
    """Keep unparsed sources distinct from verified zero-elector sources."""
    return (
        [],
        [],
        {
            "filename": part["filename"],
            "source_sha256": part["source_sha256"],
            "split": part["split"],
            "parse_status": status,
            "counts": dict.fromkeys(
                ["events", "inventory", "active_total", "active_assembly", "active_npr"]
            ),
            "issues": issues,
            "printed_final_total": part.get("printed_final_total"),
            "final_total_difference": None,
        },
    )


def source_id(row):
    """Recover a unique ASCII printed identifier from the first card text line."""
    if row["id"]:
        return row["id"]
    lines = (row["latin_text"] or "").splitlines()
    candidates = {
        token.upper()
        for token in (lines[0].split() if lines else [])
        if re.fullmatch(r"[A-Za-z]{2,5}[0-9/]{6,12}", token)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def reconcile_part(rows, part, labels):
    """Apply the identity ledger only when all required sections are recognized."""
    pages = {p["page"]: p for p in labels}
    summaries = part.get("closing_summaries", [])
    identity = part.get("part_identity", {})
    printed = (
        summaries[0]
        if len(summaries) == 1
        and summaries[0].get("format_supported")
        and identity.get("valid") is not False
        else {}
    )
    seen_components = {
        m["component"] for p in labels for m in p["markers"] if m["label"] is not None
    }
    required = {
        component
        for component, kind in [(1, "addition"), (2, "deletion"), (3, "correction")]
        if printed.get(kind + "_total", 0) > 0
    }
    issues = [
        {"reason": "required_section_heading_missing", "component": component}
        for component in sorted(required - seen_components)
    ]
    if identity.get("valid") is False:
        issues.append({"reason": "printed_part_identity_mismatch"})
    events = []
    recovered_ids = 0
    for row in rows:
        label = pages[row["page"]]
        if label["event_type"] is None:
            issues.append(
                {"reason": "unclassified_event", "event_key": row["appearance_key"]}
            )
        identifier = source_id(row)
        recovered_ids += row["id"] is None and identifier is not None
        events.append(
            {
                **row,
                "id": identifier,
                "event_key": row["appearance_key"],
                "event_type": label["event_type"],
                "assembly_eligible": label["assembly_eligible"],
                "relative_issue": "urdu_text_not_verified",
                "deletion_stamp_raw": row["stamp_candidate"],
            }
        )
    counts = Counter(r["event_type"] or "unknown" for r in events)
    inventory = []
    status = (
        "unresolved_source"
        if identity.get("valid") is False
        else "unresolved_sections"
        if issues
        else "reconciled"
    )
    if not issues:
        try:
            inventory, ledger_issues = reconcile(events)
            issues.extend(ledger_issues)
        except ValueError as error:
            issues.append({"reason": "unresolved_identity", "detail": str(error)})
            status = "unresolved_identity"
    active = sum(r["active"] for r in inventory) if status == "reconciled" else None
    summary = {
        "filename": part["filename"],
        "source_sha256": part["source_sha256"],
        "split": part["split"],
        "parse_status": status,
        "counts": {
            "events": len(events),
            "inventory": len(inventory),
            "active_total": active,
            "active_assembly": sum(
                r["active"] and r["assembly_eligible"] for r in inventory
            )
            if active is not None
            else None,
            "active_npr": sum(
                r["active"] and not r["assembly_eligible"] for r in inventory
            )
            if active is not None
            else None,
        },
        "event_counts": dict(counts),
        "case_normalized_ids_recovered": recovered_ids,
        "printed_final_total": printed.get("final_total"),
        "final_total_difference": active - printed["final_total"]
        if active is not None and printed
        else None,
        "event_count_differences": {
            kind: counts[kind] - printed[kind + "_total"]
            for kind in ("base", "addition", "deletion", "correction")
        }
        if printed
        else None,
        "page_sequence": part["page_sequence"],
        "part_identity": identity,
        "section_labels": labels,
        "issues": issues,
    }
    return events, inventory, summary


def inspect_part(payload, pdf_dir, catalog):
    """Verify source bytes and interpret the saved header glyph evidence."""
    import pymupdf

    from model_training.prep_er_data.jk_urdu_audit import inspect_part_identity

    rows, part = payload
    path = pdf_dir / part["filename"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != part["source_sha256"]:
        raise ValueError("Source PDF hash changed")
    with pymupdf.open(path) as document:
        fonts = HeaderFonts(document)
        outlines = {
            p["page"]: fonts.page(document[p["page"] - 1], p["header_spans"])
            for p in part["pages"]
        }
        labels = classify_pages(part["pages"], outlines, catalog)
        part = {
            **part,
            "part_identity": inspect_part_identity(
                document, part["filename"], part.get("closing_summaries", [])
            ),
        }
    return reconcile_part(rows, part, labels)


def isolated_part(payload, pdf_dir, catalog_path, timeout):
    """Discard incomplete worker output on a timeout or parser error."""
    if not payload[0]:
        return unavailable_part(
            payload[1], "no_structural_rows", payload[1].get("issues", [])
        )
    with tempfile.TemporaryDirectory(prefix="jk-urdu-ledger-") as temporary:
        directory = Path(temporary)
        source, result = directory / "input.json", directory / "result.json"
        source.write_text(json.dumps(payload))
        command = [
            sys.executable,
            "-m",
            "model_training.prep_er_data.jk_urdu_rows",
            "--inspect",
            str(source),
            "--result",
            str(result),
            "--pdf-dir",
            str(pdf_dir),
            "--catalog",
            str(catalog_path),
        ]
        try:
            subprocess.run(  # noqa: S603 -- fixed module, paths passed as arguments
                command, check=True, timeout=timeout, capture_output=True
            )
            return json.loads(result.read_text())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            return unavailable_part(
                payload[1], "worker_failure", [{"reason": type(error).__name__}]
            )


def main():
    """Stream bounded batches into typed events, inventory and a source audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-dir", type=Path)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument(
        "--parts", type=Path, help="JSON list of source basenames to rerun"
    )
    parser.add_argument("--inspect", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    if args.inspect:
        args.result.write_text(
            json.dumps(
                inspect_part(
                    json.loads(args.inspect.read_text()), args.pdf_dir, catalog
                )
            )
        )
        return
    if (
        not args.structure_dir
        or not args.out_dir
        or args.workers < 1
        or args.timeout <= 0
    ):
        parser.error(
            "Require structural input, a new output directory "
            "and positive worker/timeout values"
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    audit_path = args.structure_dir / "audit.json"
    if not audit_path.exists():
        audit_path = args.structure_dir / "audit.json.gz"
    with open_audit(audit_path) as stream:
        artifacts = next(ijson.items(stream, "artifacts", use_float=True))
    source_path = args.structure_dir / "appearances.parquet"
    if file_sha256(source_path) != artifacts["appearances.parquet"]:
        raise ValueError("Structural appearance artifact hash changed")
    selected = None
    if args.parts:
        selected = json.loads(args.parts.read_text())
        if (
            not isinstance(selected, list)
            or not selected
            or not all(isinstance(name, str) for name in selected)
            or len(set(selected)) != len(selected)
        ):
            parser.error(
                "--parts must contain a nonempty JSON list of unique filenames"
            )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    output = output_schema()
    records = (
        r
        for b in pq.ParquetFile(source_path).iter_batches(batch_size=10000)
        for r in b.to_pylist()
    )
    payloads = select_parts(structural_payloads(audit_path, records), selected)
    audits, seen = [], set()
    with (
        pq.ParquetWriter(
            args.out_dir / "events.parquet", output, compression="zstd"
        ) as event_writer,
        pq.ParquetWriter(
            args.out_dir / "inventory.parquet", output, compression="zstd"
        ) as inventory_writer,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        for batch in itertools.batched(payloads, args.workers * 2):
            futures = [
                pool.submit(isolated_part, p, args.pdf_dir, args.catalog, args.timeout)
                for p in batch
            ]
            for future in futures:
                events, inventory, part = future.result()
                name = part["filename"]
                if name in seen:
                    raise ValueError("Structural source rows are not contiguous by PDF")
                seen.add(name)
                audits.append(part)
                event_writer.write_table(pa.Table.from_pylist(events, schema=output))
                inventory_writer.write_table(
                    pa.Table.from_pylist(inventory, schema=output)
                )
                emit(
                    {
                        k: part[k]
                        for k in (
                            "filename",
                            "parse_status",
                            "counts",
                            "final_total_difference",
                        )
                    }
                )
    schema = {
        "fields": [
            {"name": f.name, "type": str(f.type), "nullable": f.nullable}
            for f in output
        ],
        "names": (
            "Names and relative names remain null; "
            "no native Urdu transcription is accepted."
        ),
    }
    (args.out_dir / "SCHEMA.json").write_text(json.dumps(schema, indent=2) + "\n")
    summary = {
        "source_parts": len(audits),
        "status_counts": dict(Counter(p["parse_status"] for p in audits)),
        "inventory_rows": sum(p["counts"]["inventory"] or 0 for p in audits),
        "active_total": sum(p["counts"]["active_total"] or 0 for p in audits),
        "comparable_parts": sum(
            p["final_total_difference"] is not None for p in audits
        ),
        "matching_parts": sum(p["final_total_difference"] == 0 for p in audits),
        "absolute_difference": sum(
            abs(p["final_total_difference"] or 0) for p in audits
        ),
        "accepted_urdu_names": 0,
    }
    audit = {
        "summary": summary,
        "parts": audits,
        "structure_audit_sha256": file_sha256(audit_path),
        "structure_audit_file": audit_path.name,
        "selected_parts": selected,
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "script_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in [
                "jk_urdu_rows.py",
                "jk_urdu_sections.py",
                "jk_urdu_audit.py",
                "jk_hindi_rows.py",
            ]
        },
        "artifacts": {
            name: file_sha256(args.out_dir / name)
            for name in ["events.parquet", "inventory.parquet", "SCHEMA.json"]
        },
        "limitations": [
            "Names remain unavailable; artifacts establish source accounting only.",
            "Unknown required headings and unresolved identity collisions "
            "withhold whole parts.",
            "Printed control differences and incomplete page sequences "
            "remain explicit source/parser exceptions.",
            "Hindi/Urdu editions can overlap; do not add their record totals.",
        ],
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    emit(summary)


if __name__ == "__main__":
    main()
