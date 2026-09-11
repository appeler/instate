"""Rebuild J&K English PDFs with source hashes and explicit roll eligibility.

Uses the adjacent parse_searchable_rolls reader. Reads local PDFs only; the
output retains source text, blank names, NPR rows, and per-part discrepancies.
"""

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FIELDS = (
    "record_key filename number id elector_name father_or_husband_name "
    "relative_type house_no age sex ac_name part_no year raw_cell"
).split()
SCHEMA = pa.schema(
    [(name, pa.string()) for name in FIELDS]
    + [("source_name_missing", pa.bool_()), ("assembly_eligible", pa.bool_())]
)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse(path, reader):
    audit = {"filename": path.name, "source_sha256": sha256(path)}
    try:
        roll = reader(str(path))
    except Exception as exc:
        return {**audit, "status": "reader_error", "error": str(exc)}, []
    rows = []
    for ordinal, elector in enumerate(roll.electors.values(), 1):
        values = [
            f"{path.name}:{ordinal}",
            path.name,
            elector.number,
            elector.id,
            elector.name,
            elector.relative_name,
            elector.find("relativeType"),
            elector.house,
            elector.age,
            elector.sex,
            roll.general.ac_name,
            roll.general.part_no,
            roll.general.year,
            elector.get_text(),
        ]
        rows.append(
            dict(
                zip(FIELDS, values, strict=True),
                source_name_missing=not bool(elector.name),
                assembly_eligible=elector.assembly_eligible,
            )
        )
    printed = int(roll.general.net_total) if roll.general.net_total.isdigit() else None
    return {
        **audit,
        "status": "parsed",
        "rows": len(rows),
        "printed_total": printed,
        "difference": len(rows) - printed if printed is not None else None,
        "blank_names": sum(row["source_name_missing"] for row in rows),
        "duplicate_numbers": len(rows) - len({row["number"] for row in rows}),
        "npr_rows": len(roll.npr_electors),
        "relationship_counts": dict(Counter(row["relative_type"] for row in rows)),
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--parser-root", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.pdf_dir.glob("*.pdf"))
    if not sources:
        parser.error("--pdf-dir contains no PDFs")
    parser_root = args.parser_root.resolve() / "pdfparser"
    sys.path.insert(0, str(parser_root))
    reader = importlib.import_module("modules.rolls.jk.english").JkPDF
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=False)
    parts = []
    artifact = output / "jk_english_2018.parquet"
    with pq.ParquetWriter(artifact, SCHEMA) as writer:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for audit, rows in pool.map(partial(parse, reader=reader), sources):
                parts.append(audit)
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    parsed = [part for part in parts if part["status"] == "parsed"]
    compared = [part for part in parsed if part["difference"] is not None]
    summary = {
        "source_pdfs": len(parts),
        "parsed_pdfs": len(parsed),
        "rows": sum(p["rows"] for p in parsed),
        "blank_names": sum(p["blank_names"] for p in parsed),
        "npr_rows": sum(p["npr_rows"] for p in parsed),
        "duplicate_numbers": sum(p["duplicate_numbers"] for p in parsed),
        "parts_with_printed_total": len(compared),
        "printed_total": sum(p["printed_total"] for p in compared),
        "exact_parts": sum(p["difference"] == 0 for p in compared),
        "net_difference": sum(p["difference"] for p in compared),
        "absolute_difference": sum(abs(p["difference"]) for p in compared),
    }
    manifest = {
        "summary": summary,
        "parts": parts,
        "parser_sha256": {
            name: sha256(parser_root / name)
            for name in (
                "modules/rolls/jk/english.py",
                "modules/rolls/base.py",
                "modules/pdf.py",
                "helpers/__init__.py",
            )
        },
        "script_sha256": sha256(Path(__file__)),
        "artifact_sha256": sha256(artifact),
        "schema": {field.name: str(field.type) for field in SCHEMA},
    }
    (output / "audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
