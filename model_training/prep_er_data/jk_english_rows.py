"""Rebuild the original J&K English rolls, including their supplement ledger."""

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

from model_training.prep_er_data.english_glyphs import (
    DocumentFonts,
    ReferenceFonts,
    extract_page,
)
from model_training.prep_er_data.jk_hindi_rows import (
    FIELD_DESCRIPTIONS,
    card_frames,
    contains,
    interpret_summary,
    output_schema,
    page_sequence,
    parse_serial,
    reconcile,
    text_lines,
    validate_inputs,
)


def join_lines(lines, field="candidate"):
    return "\n".join(" ".join(word[field] for word in line) for line in lines)


def accept_name(candidate):
    if not candidate:
        return None, "missing_name"
    if "�" in candidate or any(not 32 <= ord(c) < 127 for c in candidate):
        return None, "unverified_rendering"
    return candidate, None


def source_name_labels(glyphs):
    text = join_lines(text_lines([g for g in glyphs if "Bold" not in g["font"]]))
    return len(re.findall(r"\bElectors\s+Name\s*:", text))


def parse_card(box, glyphs):
    inside = [g for g in glyphs if contains(box, g["origin"])]
    body = [g for g in inside if "Bold" not in g["font"]]
    lines = text_lines(body)
    header = text_lines([g for g in body if g["origin"][1] < box[1] + 17])
    words = [word for line in header for word in line]
    serials = [
        w["candidate"]
        for w in words
        if w["x"] < box[0] + 45 and parse_serial(w["candidate"]) is not None
    ]
    ids = [
        w["candidate"]
        for w in words
        if re.fullmatch(r"[A-Z]{2,4}[0-9/]{6,12}", w["candidate"])
    ]
    row = dict.fromkeys(
        (
            "name_candidate elector_name relative_candidate relative_name "
            "relative_type house_no age sex_candidate"
        ).split()
    )
    row.update(
        number=parse_serial(serials[0]) if len(serials) == 1 else None,
        number_raw=serials[0] if len(serials) == 1 else None,
        id=ids[0] if len(ids) == 1 else None,
        header_raw=join_lines(header, "raw"),
        raw_cell=join_lines(text_lines(inside), "raw"),
        candidate_cell=join_lines(lines),
        bbox=list(box),
    )
    continuation = None
    for line in lines:
        text = " ".join(w["candidate"] for w in line)
        name = re.match(r"Electors\s+Name\s*:\s*(.*)", text)
        relative = re.match(
            r"(Father's Name|Husband's Name|Mother's Name|Self|Son in Law|"
            r"Adopted Son|Name)\s*:\s*(.*)",
            text,
        )
        house = re.match(r"House\s+No\s*:\s*(.*)", text)
        age = re.search(r"\bAge\s*:\s*([0-9]+)\b", text)
        sex = re.search(r"\bSex\s*:\s*(.*)", text)
        if name:
            continuation = "name_candidate"
            row[continuation] = name[1].strip() or None
        elif relative:
            continuation = "relative_candidate"
            row[continuation] = relative[2].strip() or None
            row["relative_type"] = relative[1]
        elif house or age or sex:
            continuation = None
            if house:
                row["house_no"] = house[1].strip() or None
            if age:
                row["age"] = age[1]
            if sex:
                row["sex_candidate"] = sex[1].strip() or None
        elif re.search(r"\b(?:Name|No|Age|Sex)\s*:", text):
            continuation = None
        elif continuation:
            row[continuation] = " ".join(
                value for value in (row[continuation], text.strip()) if value
            )
    for prefix, accepted, issue in (
        ("name", "elector_name", "name_issue"),
        ("relative", "relative_name", "relative_issue"),
    ):
        row[accepted], row[issue] = accept_name(row[prefix + "_candidate"])
    stamp = "".join(
        g["decoded"]
        for g in sorted(inside, key=lambda g: g["origin"][0])
        if "Bold" in g["font"]
    )
    row["deletion_stamp_raw"] = stamp or None
    row["deleted_stamp"] = stamp == "DELETED"
    return row if row["number"] or row["id"] or row["name_candidate"] else None


def component_type(text):
    titles = {"I": ("Addition", "addition"), "II": ("Deletion", "deletion")}
    titles["III"] = ("Correction", "correction")
    match = re.search(r"\bComponent\s+(III|II|I)\s*-\s*(\w+)\s+List\b", text)
    if match and titles[match[1]][0] == match[2]:
        return titles[match[1]][1]
    return None


def npr_heading(text):
    return bool(
        re.search(
            r"\(\s*NPR\s*-?\s*NOT\s+ELIGIBLE\s+TO\s+VOTE\s+IN\s+ASSEMBLY",
            text,
        )
    )


def closing_summary(page, lines):
    text = join_lines(lines)
    if "SUMMARY OF ELECTORS" not in text or "(I+II-III)" not in text:
        return None
    numeric_rows = []
    for line in lines:
        values = [
            w
            for w in line
            if w["x"] > page.rect.width * 0.67
            and parse_serial(w["candidate"]) is not None
        ]
        if len(values) == 3:
            male, female, total = [int(parse_serial(w["candidate"])) for w in values]
            numeric_rows.append(
                {"y": values[0]["y"], "male": male, "female": female, "total": total}
            )
    return {
        "page": page.number + 1,
        "numeric_rows": numeric_rows,
        "candidate_text": text,
        **interpret_summary(numeric_rows),
    }


def parse_pdf(path, references):
    import pymupdf

    events, pages, summaries, issues = [], [], [], []
    kind, eligible, supplement = "base", True, None
    with pymupdf.open(path) as document:
        fonts = DocumentFonts(document, references)
        physical_count = len(document)
        for page in document:
            if page.number < 2:
                continue
            glyphs = extract_page(page, fonts)
            lines = text_lines(glyphs)
            candidate = join_lines(lines)
            component = component_type(candidate)
            if component:
                kind = component
                eligible = not npr_heading(candidate)
            elif kind == "base" and npr_heading(candidate):
                eligible = False
            elif "Component" in candidate:
                issues.append(
                    {"reason": "unreadable_component_heading", "page": page.number + 1}
                )
                kind = "unknown"
            supplementary = re.search(r"Supplement No\.\s*:\s*([0-9]+)", candidate)
            if supplementary:
                supplement = supplementary[1]
            summary = closing_summary(page, lines)
            if summary:
                summaries.append(summary)
            rectangles = [
                tuple(item[1])
                for drawing in page.get_drawings()
                for item in drawing["items"]
                if item[0] == "re"
            ]
            frames = card_frames(rectangles)
            rows = []
            for box in frames:
                row = parse_card(box, glyphs)
                if row:
                    row.update(
                        filename=path.name,
                        page=page.number + 1,
                        event_key=f"{path.name}:{page.number + 1}:{len(rows) + 1}",
                        event_type=kind,
                        assembly_eligible=eligible,
                        supplement=supplement,
                    )
                    rows.append(row)
            labels = source_name_labels(glyphs)
            if labels != len(rows):
                issues.append(
                    {
                        "reason": "name_label_event_mismatch",
                        "page": page.number + 1,
                        "labels": labels,
                        "events": len(rows),
                    }
                )
            events.extend(rows)
            footer = join_lines(
                text_lines(
                    [
                        g
                        for g in glyphs
                        if g["origin"][0] > page.rect.width * 0.75
                        and g["origin"][1] > page.rect.height - 55
                    ]
                )
            )
            printed_page = re.search(r"\b([0-9]+)\s+of\s+([0-9]+)\b", footer)
            pages.append(
                {
                    "page": page.number + 1,
                    "printed_page": int(printed_page[1]) if printed_page else None,
                    "printed_page_total": int(printed_page[2])
                    if printed_page
                    else None,
                    "frames": len(frames),
                    "source_name_labels": labels,
                    "events": len(rows),
                    "event_type": kind,
                    "assembly_eligible": eligible,
                    "glyph_status": dict(Counter(g["status"] for g in glyphs)),
                }
            )
    inventory, ledger_issues = reconcile(events)
    issues.extend(ledger_issues)
    sequence = page_sequence(pages, physical_count)
    if sequence["valid"] is False:
        issues.append({"reason": "source_page_sequence_mismatch", **sequence})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in events + inventory:
        row["source_sha256"] = digest
    printed = summaries[0] if len(summaries) == 1 else {}
    comparisons = {}
    if printed.get("format_supported"):
        for event_type in ("base", "addition", "deletion", "correction"):
            comparisons[event_type] = (
                sum(r["event_type"] == event_type for r in events)
                - printed[event_type + "_total"]
            )
        comparisons["active"] = (
            sum(r["active"] for r in inventory) - printed["final_total"]
        )
    return (
        events,
        inventory,
        {
            "filename": path.name,
            "source_sha256": digest,
            "parse_status": "parsed" if events else "no_events",
            "event_counts": dict(
                Counter(
                    ("assembly_" if row["assembly_eligible"] else "npr_")
                    + row["event_type"]
                    for row in events
                )
            ),
            "events": len(events),
            "inventory": len(inventory),
            "active_assembly": sum(
                r["active"] and r["assembly_eligible"] for r in inventory
            ),
            "active_npr": sum(
                r["active"] and not r["assembly_eligible"] for r in inventory
            ),
            "active_total": sum(r["active"] for r in inventory),
            "accepted_names": sum(
                r["elector_name"] is not None for r in inventory if r["active"]
            ),
            "deletion_stamps": sum(r["deleted_stamp"] for r in events),
            "printed_final_total": printed.get("final_total"),
            "printed_arithmetic_valid": printed.get("arithmetic_valid", False),
            "count_differences": comparisons,
            "fully_reconciled": bool(comparisons)
            and not any(comparisons.values())
            and not issues
            and printed.get("arithmetic_valid", False),
            "issues": issues,
            "pages": pages,
            "page_sequence": sequence,
            "closing_summaries": summaries,
        },
    )


_REFERENCES = None


def initialize_worker(paths):
    global _REFERENCES
    _REFERENCES = ReferenceFonts(paths)


def parse_job(path):
    if _REFERENCES is None:
        raise RuntimeError("Worker reference fonts are not initialized")
    # Fail the batch on unexpected exceptions; never publish partial success.
    return parse_pdf(path, _REFERENCES)


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--reference-font", type=Path, action="append", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    split = json.loads(args.split.read_text())
    paths = validate_inputs(split, [args.pdf_dir])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started, schema, parts = time.monotonic(), output_schema(), []
    jobs = [(label, name) for label, names in split.items() for name in names]
    with ExitStack() as stack:
        outputs = [
            stack.enter_context(
                pq.ParquetWriter(args.out_dir / name, schema, compression="zstd")
            )
            for name in ("events.parquet", "inventory.parquet")
        ]
        if args.workers == 1:
            initialize_worker(args.reference_font)
            results = map(parse_job, (paths[name] for _, name in jobs))
        else:
            executor = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=args.workers,
                    initializer=initialize_worker,
                    initargs=(args.reference_font,),
                )
            )
            results = executor.map(parse_job, (paths[name] for _, name in jobs))
        for (label, _), (events, inventory, audit) in zip(jobs, results, strict=True):
            audit["split"] = label
            parts.append(audit)
            for rows, output in zip((events, inventory), outputs, strict=True):
                output.write_table(pa.Table.from_pylist(rows, schema=schema))
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in audit.items()
                        if key not in ("pages", "issues", "closing_summaries")
                    }
                ),
                flush=True,
            )
    descriptions = {
        **FIELD_DESCRIPTIONS,
        "elector_name": "Name whose ASCII glyphs all passed reference contour checks.",
        "relative_type": "Literal English relationship label, without inference.",
        "raw_cell": "Position-ordered source text inside the card, before decoding.",
        "deletion_stamp_raw": "Position-ordered decoded bold glyphs inside the card.",
    }
    (args.out_dir / "SCHEMA.json").write_text(
        json.dumps(
            {
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                        "description": descriptions[field.name],
                    }
                    for field in schema
                ]
            },
            indent=2,
        )
        + "\n"
    )
    summary = {
        "pdfs": len(parts),
        **{
            key: sum(part[key] for part in parts)
            for key in (
                "events",
                "inventory",
                "active_total",
                "active_assembly",
                "active_npr",
                "accepted_names",
                "fully_reconciled",
            )
        },
        "parts_matching_final_total": sum(
            part["count_differences"].get("active") == 0 for part in parts
        ),
        "parts_with_issues": sum(bool(part["issues"]) for part in parts),
    }
    audit = {
        "summary": summary,
        "roll_year": 2018,
        "geography": "Historical J&K AC047-050, present-day Ladakh",
        "parts": parts,
        "reference_fonts": [
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.reference_font
        ],
        "script_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "jk_english_rows.py",
                "english_glyphs.py",
                "jk_hindi_rows.py",
                "hindi_text.py",
                "hindi_glyphs.py",
            )
        },
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "artifacts": {
            name: hashlib.sha256((args.out_dir / name).read_bytes()).hexdigest()
            for name in ("events.parquet", "inventory.parquet", "SCHEMA.json")
        },
        "versions": {
            name: version(name)
            for name in ("PyMuPDF", "fonttools", "pillow", "pyarrow")
        },
        "elapsed_seconds": time.monotonic() - started,
        "workers": args.workers,
        "limitations": [
            "Contour matching uses a 0.004-em tolerance; it is not name ground truth.",
            "ASCII only; unsupported and ambiguous outlines remain explicit.",
            "Serials are keyed separately for assembly and NPR records.",
            "Cleaned archive totals belong to another source version.",
            "Source roll calls its Urdu version authoritative.",
            "Surname and released model artifacts are unchanged.",
        ],
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
