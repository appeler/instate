"""Audit Urdu roll boxes and closing totals without accepting damaged names.

Each PDF is inspected in an isolated process with a timeout. This is a source
inventory and control-total audit, not a reconciled elector or surname dataset.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LATIN_FONTS = {
    "TimesNewRomanPSMT",
    "TimesNewRomanPS-BoldMT",
    "ArialMT",
    "Arial-BoldMT",
    "Times-Roman",
    "Times-Bold",
    "Times-Italic",
    "Times-BoldItalic",
    "Helvetica",
    "Helvetica-Bold",
    "Helvetica-Oblique",
    "Helvetica-BoldOblique",
}


def latin_glyphs(spans):
    """Retain printable glyphs from supported Latin fonts only."""
    from model_training.prep_er_data.hindi_glyphs import glyph_groups

    glyphs = []
    for span in spans:
        if span["font"].split("+")[-1] not in LATIN_FONTS:
            continue
        for group in glyph_groups(span["chars"]):
            decoded = group["text"]
            if not all(32 <= ord(char) < 127 for char in decoded):
                decoded = "�"
            glyphs.append(
                {
                    **group,
                    "decoded": decoded,
                    "font": span["font"],
                    "ordinal": len(glyphs),
                }
            )
    return glyphs


def joined(glyphs):
    """Join spatially ordered Latin glyphs into source text lines."""
    from model_training.prep_er_data.jk_hindi_rows import text_lines

    return "\n".join(
        " ".join(word["candidate"] for word in line) for line in text_lines(glyphs)
    )


def printed_part(glyphs, *, closing=False):
    """Read the part identifier from the fixed Latin header column only."""
    low, high = (35, 65) if closing else (75, 99)
    text = joined(
        [g for g in glyphs if 30 < g["origin"][0] < 90 and low < g["origin"][1] < high]
    )
    match = re.fullmatch(r"\s*([0-9]{1,4})\s*(M)?\s*", text)
    return f"{int(match[1])}{match[2] or ''}" if match else None


def part_identity(filename, body, closing):
    """Separate inconsistent printed headers from a differing archive basename."""
    match = re.fullmatch(r"UACA[0-9]{3}PS([0-9]{4})\.pdf", filename)
    observations = [p["part"] for p in body + closing if p["part"] is not None]
    expected = int(match[1]) if match else None
    mismatch = len(set(observations)) > 1
    filename_matches = (
        all(int(value.rstrip("M")) == expected for value in observations)
        if expected is not None and observations
        else None
    )
    complete = len(body) == 2 and len(closing) == 1 and len(observations) == 3
    return {
        "filename_part": expected,
        "filename_matches_printed_part": filename_matches,
        "body_headers": body,
        "closing_headers": closing,
        "valid": False if mismatch else True if complete else None,
        "reason": "printed_part_identity_mismatch"
        if mismatch
        else None
        if complete
        else "part_header_evidence_incomplete",
    }


def inspect_part_identity(document, filename, summaries):
    """Inspect header identities without reading names or trusting closing totals."""
    closing_pages = {p["page"] for p in summaries}
    body = [
        {
            "page": number,
            "part": printed_part(latin_glyphs(document[number - 1].get_texttrace())),
        }
        for number in (3, 4)
        if number <= len(document) and number not in closing_pages
    ]
    closing = [
        {
            "page": number,
            "part": printed_part(
                latin_glyphs(document[number - 1].get_texttrace()), closing=True
            ),
        }
        for number in sorted(closing_pages)
    ]
    return part_identity(filename, body, closing)


def closing_summary(page, glyphs, *, last_page=False, following_date=False):
    """Read the supported closing-table layouts and check their arithmetic."""
    from model_training.prep_er_data.jk_hindi_rows import (
        interpret_summary,
        parse_serial,
        text_lines,
    )

    text = joined(glyphs)
    formula = bool(re.search(r"I\s*\+\s*II\s*-\s*III", text))
    dated_last_page = last_page and (
        following_date or bool(re.search(r"\b[0-9]{2}-[A-Z][a-z]{2}-[0-9]{4}\b", text))
    )
    if not formula and not dated_last_page:
        return None
    rows = []
    for line in text_lines(glyphs):
        words = [
            w
            for w in line
            if w["x"] < page.rect.width * 0.37
            and parse_serial(w["candidate"]) is not None
        ]
        if len(words) == 3:
            total, female, male = [int(parse_serial(w["candidate"])) for w in words]
            rows.append(
                {"y": words[0]["y"], "male": male, "female": female, "total": total}
            )
    if not formula and len(rows) != 8:
        return None
    return {
        "page": page.number + 1,
        "recognition": "formula" if formula else "dated_closing_eight_row_table",
        "numeric_rows": rows,
        **interpret_summary(rows),
    }


def parse_card(box, glyphs):
    """Keep card serials, identifiers and deletion stamps without accepting names."""
    from model_training.prep_er_data.jk_hindi_rows import (
        contains,
        parse_serial,
        text_lines,
    )

    inside = [g for g in glyphs if contains(box, g["origin"])]
    header = [
        g for g in inside if g["origin"][1] < box[1] + 20 and "Bold" not in g["font"]
    ]
    words = [w for line in text_lines(header) for w in line]
    serials = [
        w["candidate"]
        for w in words
        if w["x"] < box[0] + 50 and parse_serial(w["candidate"]) is not None
    ]
    identifiers = [
        w["candidate"]
        for w in words
        if re.fullmatch(r"[A-Z]{2,5}[0-9/]{6,12}", w["candidate"])
    ]
    if not serials and not identifiers:
        return None
    stamp = "".join(
        g["decoded"]
        for g in sorted(inside, key=lambda g: g["origin"][0])
        if "Bold" in g["font"]
    )
    return {
        "number": parse_serial(serials[0]) if len(serials) == 1 else None,
        "number_raw": serials[0] if len(serials) == 1 else None,
        "id": identifiers[0] if len(identifiers) == 1 else None,
        "bbox": list(box),
        "latin_text": joined(inside),
        "deleted_stamp": stamp == "DELETED",
        "stamp_candidate": stamp or None,
        "event_type": None,
        "assembly_eligible": None,
        "elector_name": None,
        "name_issue": "urdu_text_not_verified",
    }


def inspect_pdf(path, *, closing_only=False, structure_only=False):
    """Inspect source counts, closing controls and printed part identities."""
    import pymupdf

    from model_training.prep_er_data.jk_hindi_rows import card_frames, page_sequence

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, pages, summaries, fonts, issues = [], [], [], set(), []
    with pymupdf.open(path) as document:
        physical_pages = len(document)
        if not physical_pages:
            rows, audit = failed_audit(
                path, "source_has_no_pages", "The PDF reader found no accessible pages"
            )
            audit.update(physical_pages=0, pdf_repaired=document.is_repaired)
            return rows, audit
        last_glyphs = latin_glyphs(document[-1].get_texttrace())
        final_date = bool(
            re.search(r"\b[0-9]{2}-[A-Z][a-z]{2}-[0-9]{4}\b", joined(last_glyphs))
        )
        selected_pages = (
            document[max(0, physical_pages - 2) :] if closing_only else document
        )
        for page in selected_pages:
            # Font resource traversal is deliberately inside the per-PDF timeout.
            fonts.update((f[1], f[2], f[3]) for f in page.get_fonts(full=True))
            spans = page.get_texttrace()
            glyphs = (
                last_glyphs
                if page.number == physical_pages - 1
                else latin_glyphs(spans)
            )
            text_page = None if closing_only or structure_only else page.get_textpage()
            rectangles = [
                tuple(item[1])
                for drawing in ([] if closing_only else page.get_drawings())
                for item in drawing["items"]
                if item[0] == "re"
            ]
            frames = [] if closing_only else card_frames(rectangles)
            page_rows = []
            for frame in frames:
                row = parse_card(frame, glyphs)
                if row:
                    row.update(
                        filename=path.name,
                        source_sha256=digest,
                        page=page.number + 1,
                        appearance_key=(
                            f"{path.name}:{page.number + 1}:{len(page_rows) + 1}"
                        ),
                        raw_cell=None
                        if structure_only
                        else page.get_textbox(pymupdf.Rect(frame), textpage=text_page),
                    )
                    page_rows.append(row)
            rows.extend(page_rows)
            summary = closing_summary(
                page,
                glyphs,
                last_page=page.number >= physical_pages - 2,
                following_date=final_date and page.number == physical_pages - 2,
            )
            if summary:
                summaries.append(summary)
            footer = joined(
                [
                    g
                    for g in glyphs
                    if g["origin"][0] > page.rect.width * 0.75
                    and g["origin"][1] > page.rect.height - 55
                ]
            )
            match = re.search(r"\b([0-9]+)\s+of\s+([0-9]+)\b", footer)
            pages.append(
                {
                    "page": page.number + 1,
                    "frames": len(frames),
                    "appearances": len(page_rows),
                    "printed_page": int(match[1]) if match else None,
                    "printed_page_total": int(match[2]) if match else None,
                }
            )
            if structure_only:
                header_bottom = min([330, *(frame[1] for frame in frames)])
                pages[-1]["header_spans"] = [
                    {
                        "font": span["font"],
                        "chars": [
                            [char[0], char[1], *char[2]]
                            for char in span["chars"]
                            if 55 < char[2][1] < header_bottom
                        ],
                    }
                    for span in spans
                    if any(55 < char[2][1] < header_bottom for char in span["chars"])
                ]
        sequence = (
            {
                "valid": None,
                "reason": "closing_pages_only",
                "physical_pages": physical_pages,
            }
            if closing_only
            else page_sequence(pages[2:], physical_pages)
        )
        identity = inspect_part_identity(document, path.name, summaries)
    if identity["valid"] is False:
        issues.append({"reason": identity["reason"]})
    if identity["filename_matches_printed_part"] is False:
        issues.append({"reason": "source_filename_part_mismatch"})
    if sequence["valid"] is False:
        issues.append({"reason": "source_page_sequence_mismatch", **sequence})
    missing_serial = sum(r["number"] is None for r in rows)
    missing_id = sum(r["id"] is None for r in rows)
    if missing_serial:
        issues.append(
            {"reason": "missing_or_ambiguous_serial", "appearances": missing_serial}
        )
    if missing_id:
        issues.append({"reason": "missing_or_ambiguous_id", "appearances": missing_id})
    if not rows and not closing_only:
        issues.append({"reason": "no_numbered_cards"})
    printed = summaries[0] if len(summaries) == 1 else {}
    if not printed.get("format_supported"):
        issues.append({"reason": "closing_summary_unavailable_or_unsupported"})
    return rows, {
        "filename": path.name,
        "source_sha256": digest,
        "status": "inspected",
        "scope": "closing_pages_only" if closing_only else "all_pages",
        "raw_card_text_retained": not (closing_only or structure_only),
        "physical_pages": physical_pages,
        "appearances": None if closing_only else len(rows),
        "deletion_stamps": None
        if closing_only
        else sum(r["deleted_stamp"] for r in rows),
        "missing_serial": None if closing_only else missing_serial,
        "missing_id": None if closing_only else missing_id,
        "printed_final_total": printed.get("final_total"),
        "printed_arithmetic_valid": printed.get("arithmetic_valid", False),
        "part_identity": identity,
        "closing_control_usable": printed.get("format_supported", False)
        and identity["valid"] is True,
        "page_sequence": sequence,
        "pages": pages,
        "closing_summaries": summaries,
        "fonts": [
            {"format": ext, "type": kind, "name": name}
            for ext, kind, name in sorted(fonts)
        ],
        "issues": issues,
    }


def failed_audit(path, status, detail):
    """Preserve the source hash and failure without reporting a verified zero."""
    return [], {
        "filename": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": status,
        "appearances": 0,
        "deletion_stamps": 0,
        "missing_serial": None,
        "missing_id": None,
        "printed_final_total": None,
        "printed_arithmetic_valid": False,
        "issues": [{"reason": status, "detail": detail}],
        "pages": [],
        "fonts": [],
        "closing_summaries": [],
    }


def isolated_job(path, timeout, *, closing_only=False, structure_only=False):
    """Run one source inspection with a timeout and discard partial results."""
    with tempfile.TemporaryDirectory(prefix="jk-urdu-") as temporary:
        result = Path(temporary) / "result.json"
        command = [
            sys.executable,
            "-m",
            "model_training.prep_er_data.jk_urdu_audit",
            "--inspect",
            str(path),
            "--result",
            str(result),
        ]
        if closing_only:
            command.append("--closing-only")
        if structure_only:
            command.append("--structure-only")
        try:
            subprocess.run(  # noqa: S603 -- fixed module and separate path arguments
                command, check=True, timeout=timeout, capture_output=True
            )
        except subprocess.TimeoutExpired:
            return failed_audit(
                path,
                "inspection_timeout",
                f"Exceeded {timeout} seconds; partial results discarded",
            )
        except subprocess.CalledProcessError as error:
            return failed_audit(
                path, "inspection_error", error.stderr.decode(errors="replace")[-2000:]
            )
        return json.loads(result.read_text())


def schema():
    """Describe the typed appearance artifact retained by the source audit."""
    import pyarrow as pa

    strings = [
        "appearance_key",
        "filename",
        "source_sha256",
        "number",
        "number_raw",
        "id",
        "latin_text",
        "raw_cell",
        "stamp_candidate",
        "event_type",
        "elector_name",
        "name_issue",
    ]
    return pa.schema(
        [(name, pa.string()) for name in strings]
        + [
            ("page", pa.int32()),
            ("bbox", pa.list_(pa.float64())),
            ("deleted_stamp", pa.bool_()),
            ("assembly_eligible", pa.bool_()),
        ]
    )


def main():
    """Run bounded source inspections and write their artifacts and audit."""
    from functools import partial
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--closing-only",
        action="store_true",
        help="Inspect closing totals and two body part headers; omit appearances",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Keep card count evidence and header glyphs; omit damaged raw card text",
    )
    parser.add_argument("--inspect", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inspect:
        if not args.result:
            parser.error("--result is required with --inspect")
        args.result.write_text(
            json.dumps(
                inspect_pdf(
                    args.inspect,
                    closing_only=args.closing_only,
                    structure_only=args.structure_only,
                )
            )
        )
        return
    if not all((args.pdf_dir, args.split, args.out_dir)):
        parser.error("--pdf-dir, --split and --out-dir are required")
    if args.workers < 1 or args.timeout <= 0:
        parser.error("--workers and --timeout must be positive")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from model_training.prep_er_data.jk_hindi_rows import validate_inputs

    split = json.loads(args.split.read_text())
    paths = validate_inputs(split, [args.pdf_dir])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started, parts, output_schema = time.monotonic(), [], schema()
    jobs = [(label, name) for label, names in split.items() for name in names]
    with (
        pq.ParquetWriter(
            args.out_dir / "appearances.parquet", output_schema, compression="zstd"
        ) as writer,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        results = pool.map(
            partial(
                isolated_job,
                timeout=args.timeout,
                closing_only=args.closing_only,
                structure_only=args.structure_only,
            ),
            (paths[name] for _, name in jobs),
        )
        for (label, _), (rows, part) in zip(jobs, results, strict=True):
            part["split"] = label
            parts.append(part)
            writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
            print(  # noqa: T201 -- command-line progress records
                json.dumps(
                    {
                        k: v
                        for k, v in part.items()
                        if k not in ("pages", "fonts", "closing_summaries", "issues")
                    }
                ),
                flush=True,
            )
    descriptions = {
        "appearance_key": "PDF basename, page and box ordinal.",
        "filename": "Source PDF basename, interpreted with its SHA-256.",
        "source_sha256": "SHA-256 of the source PDF bytes.",
        "number": "Latin-font header serial; valid grouping commas removed.",
        "number_raw": "Unnormalized serial; null when missing or ambiguous.",
        "id": "EPIC-like identifier from the Latin-font header; null when ambiguous.",
        "latin_text": "Position-ordered supported Latin text inside the box.",
        "raw_cell": (
            "Uncorrected box text, or null in structural mode; "
            "Urdu mappings may be damaged."
        ),
        "stamp_candidate": "Position-ordered bold Latin glyphs in the box.",
        "deleted_stamp": "Literal DELETED stamp; no ledger has been applied.",
        "event_type": "Always null: event classification is unverified.",
        "assembly_eligible": "Always null: NPR/assembly classification is unverified.",
        "elector_name": "Always null: damaged Urdu text is not accepted as a name.",
        "name_issue": "urdu_text_not_verified for every appearance.",
        "page": "One-based physical PDF page.",
        "bbox": "Source card bounds [x0, y0, x1, y1] in PDF points.",
    }
    (args.out_dir / "SCHEMA.json").write_text(
        json.dumps(
            {
                "fields": [
                    {
                        "name": f.name,
                        "type": str(f.type),
                        "nullable": f.nullable,
                        "description": descriptions[f.name],
                    }
                    for f in output_schema
                ]
            },
            indent=2,
        )
        + "\n"
    )
    compared = [
        p
        for p in parts
        if p["printed_final_total"] is not None and p.get("closing_control_usable")
    ]
    summary = {
        "input_pdfs": len(parts),
        "status_counts": dict(Counter(p["status"] for p in parts)),
        "appearances": None
        if args.closing_only
        else sum(p["appearances"] for p in parts),
        "deletion_stamps": None
        if args.closing_only
        else sum(p["deletion_stamps"] for p in parts),
        "parts_with_closing_totals": len(compared),
        "printed_final_total": sum(p["printed_final_total"] for p in compared),
        "parts_with_valid_closing_arithmetic": sum(
            p["printed_arithmetic_valid"] for p in compared
        ),
        "issue_counts": dict(Counter(i["reason"] for p in parts for i in p["issues"])),
    }
    audit = {
        "scope": "closing_pages_only" if args.closing_only else "all_pages",
        "raw_card_text_retained": not (args.closing_only or args.structure_only),
        "summary": summary,
        "parts": parts,
        "script_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in [
                "jk_urdu_audit.py",
                "jk_hindi_rows.py",
                "hindi_text.py",
                "hindi_glyphs.py",
            ]
        },
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "artifacts": {
            name: hashlib.sha256((args.out_dir / name).read_bytes()).hexdigest()
            for name in ["appearances.parquet", "SCHEMA.json"]
        },
        "versions": {
            name: version(name) for name in ["PyMuPDF", "pyarrow", "fonttools"]
        },
        "elapsed_seconds": time.monotonic() - started,
        "workers": args.workers,
        "timeout_seconds": args.timeout,
        "limitations": [
            "Appearances include supplements and repeats, not active elector counts.",
            "Names, event types and assembly eligibility remain unavailable.",
            "Closing totals do not establish appearance-extraction completeness.",
            "Timeouts/errors discard partial output; they are not zero-elector parts.",
            "No surname, lookup or model artifact is changed.",
        ],
    }
    with (args.out_dir / "audit.json").open("w") as stream:
        json.dump(audit, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps(summary, indent=2))  # noqa: T201 -- command-line audit summary


if __name__ == "__main__":
    main()
