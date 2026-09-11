"""Rebuild J&K Hindi elector events and reconcile printed closing summaries."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from model_training.prep_er_data.hindi_glyphs import FontOutlines, glyph_groups
from model_training.prep_er_data.hindi_text import (
    ReferenceFonts,
    extract_page,
    logical_order,
    word_groups,
)


def card_frames(rectangles):
    """Find opposing card sides; collapse duplicate strokes and prefer closed frames."""
    verticals = [
        r for r in rectangles if r[2] - r[0] <= 1.5 and 60 <= r[3] - r[1] <= 110
    ]
    horizontals = [r for r in rectangles if r[3] - r[1] <= 1.5]
    candidates = []
    for left in verticals:
        for right in verticals:
            if not 150 <= right[0] - left[0] <= 200:
                continue
            if abs(left[1] - right[1]) > 1 or abs(left[3] - right[3]) > 1:
                continue
            edges = sum(
                any(
                    abs(h[1] - y) < 1.5 and h[0] <= left[0] + 1 and h[2] >= right[0]
                    for h in horizontals
                )
                for y in (left[1], left[3])
            )
            box = (left[0], left[1], right[2], right[3])
            candidates.append((edges, box))
    boxes = []
    for _, box in sorted(
        candidates, key=lambda item: (-item[0], item[1][2] - item[1][0])
    ):

        def overlaps(other):
            intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                0, min(box[3], other[3]) - max(box[1], other[1])
            )
            area = (box[2] - box[0]) * (box[3] - box[1])
            other_area = (other[2] - other[0]) * (other[3] - other[1])
            return intersection / (area + other_area - intersection) > 0.94

        if not any(overlaps(other) for other in boxes):
            boxes.append(box)
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def contains(box, origin):
    return box[0] <= origin[0] < box[2] and box[1] <= origin[1] < box[3]


def text_lines(glyphs, references=None):
    words = []
    for word in word_groups(glyphs):
        candidate = logical_order("".join(g["decoded"] for g in word))
        verified = references is None or references.verifies(
            candidate, [g["fingerprint"] for g in word]
        )
        words.append(
            {
                "candidate": candidate,
                "raw": "".join(g["text"] for g in word),
                "verified": verified,
                "x": min(g["origin"][0] for g in word),
                "y": min(g["origin"][1] for g in word),
            }
        )
    lines = []
    for word in sorted(words, key=lambda w: (w["y"], w["x"])):
        if not lines or word["y"] - lines[-1][0]["y"] > 2.5:
            lines.append([])
        lines[-1].append(word)
    return [sorted(line, key=lambda w: w["x"]) for line in lines]


def parse_serial(text):
    """Normalize only valid plain, Western-grouped or Indian-grouped serials."""
    pattern = (
        r"(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]?(?:,[0-9]{2})+,[0-9]{3})"
    )
    return text.replace(",", "") if re.fullmatch(pattern, text) else None


def accept_name(candidate, verified):
    if not candidate:
        return None, "missing_name"
    if "◌" in candidate:
        return None, "source_placeholder"
    if not verified:
        return None, "unverified_rendering"
    return candidate, None


def parse_card(box, mangal, other, references):
    native = [g for g in mangal if contains(box, g["origin"])]
    latin = [g for g in other if contains(box, g["origin"])]
    lines = text_lines(native, references)
    header = text_lines(
        [
            g
            for g in native + latin
            if g["origin"][1] < box[1] + 20 and "Bold" not in g["font"]
        ]
    )
    header_words = [word for line in header for word in line]
    serials = [
        w["candidate"]
        for w in header_words
        if w["x"] < box[0] + 60 and parse_serial(w["candidate"]) is not None
    ]
    ids = [
        w["candidate"]
        for w in header_words
        if re.fullmatch(r"[A-Z]{2,4}[0-9/]{6,12}", w["candidate"])
    ]
    row = {
        "number": parse_serial(serials[0]) if len(serials) == 1 else None,
        "number_raw": serials[0] if len(serials) == 1 else None,
        "id": ids[0] if len(ids) == 1 else None,
        "header_raw": "\n".join(" ".join(w["raw"] for w in line) for line in header),
        "raw_cell": "\n".join(" ".join(w["raw"] for w in line) for line in lines),
        "candidate_cell": "\n".join(
            " ".join(w["candidate"] for w in line) for line in lines
        ),
        "elector_name": None,
        "name_issue": "missing_name",
        "name_candidate": None,
        "relative_name": None,
        "relative_issue": "missing_name",
        "relative_candidate": None,
        "relative_type": None,
        "house_no": None,
        "age": None,
        "sex_candidate": None,
    }
    continuation, continuation_verified = None, False
    for line in lines:
        text = " ".join(w["candidate"] for w in line)
        valid = all(w["verified"] for w in line)
        name = re.search(r"मतदाता\s*का\s*नाम\s*[:ः]\s*(.*)", text)
        if name:
            row["name_candidate"] = name[1].strip() or None
            row["elector_name"], row["name_issue"] = accept_name(
                row["name_candidate"], valid
            )
            continuation = ("name_candidate", "elector_name", "name_issue")
            continuation_verified = valid
        relative = re.search(r"(पिता|पति|माता|अन्य)\s*का\s*नाम\s*[:ः]\s*(.*)", text)
        if relative:
            row["relative_type"] = relative[1]
            row["relative_candidate"] = relative[2].strip() or None
            row["relative_name"], row["relative_issue"] = accept_name(
                row["relative_candidate"], valid
            )
            continuation = ("relative_candidate", "relative_name", "relative_issue")
            continuation_verified = valid
        house = re.search(r"मकान\s*संख्या\s*(.*)", text)
        if house:
            row["house_no"] = house[1].strip() or None
        age = re.search(r"आयु?\s*(\d+)", text)
        if age:
            row["age"] = age[1]
        sex = re.search(r"लिंग\s*(.*)", text)
        if sex:
            row["sex_candidate"] = sex[1].strip() or None
        if house or age or sex:
            continuation = None
        elif not (name or relative):
            if re.search(r"नाम\s*[:ः]|^(?:मकान|आयु?|लिंग)(?:\s|$)", text):
                continuation = None
            elif continuation:
                candidate_key, accepted_key, issue_key = continuation
                row[candidate_key] = " ".join(
                    value for value in (row[candidate_key], text.strip()) if value
                )
                continuation_verified = continuation_verified and valid
                row[accepted_key], row[issue_key] = accept_name(
                    row[candidate_key], continuation_verified
                )
    stamp = "".join(
        g["text"]
        for g in sorted(latin, key=lambda g: g["origin"][0])
        if "Bold" in g["font"]
    )
    row["deletion_stamp_raw"] = stamp or None
    row["deleted_stamp"] = stamp == "DELETED"
    if not (row["number"] or row["id"] or row["name_candidate"]):
        return None
    row["bbox"] = list(box)
    return row


def reconcile(events):
    """Keep changes as events; corrections never increase the elector count."""
    inventory, issues, deleted_keys = {}, [], set()
    for row in events:
        if not row["number"]:
            issues.append({"event_key": row["event_key"], "reason": "missing_serial"})
            continue
        key = (row["assembly_eligible"], row["number"])
        kind = row["event_type"]
        if kind in ("base", "addition"):
            if key in inventory:
                issues.append(
                    {"event_key": row["event_key"], "reason": "duplicate_entry_serial"}
                )
                continue
            inventory[key] = {
                **row,
                "active": not row["deleted_stamp"],
                "entry_event_key": row["event_key"],
                "change_event_keys": [],
            }
        elif kind in ("deletion", "correction"):
            previous = inventory.get(key)
            if previous is None:
                issues.append(
                    {"event_key": row["event_key"], "reason": "unmatched_change"}
                )
                continue
            if kind == "deletion":
                if previous["id"] and row["id"] and previous["id"] != row["id"]:
                    issues.append(
                        {
                            "event_key": row["event_key"],
                            "reason": "deletion_id_conflict",
                        }
                    )
                    continue
                if key in deleted_keys:
                    issues.append(
                        {
                            "event_key": row["event_key"],
                            "reason": "repeated_deletion_event",
                        }
                    )
                deleted_keys.add(key)
                previous["active"] = False
                previous["change_event_keys"].append(row["event_key"])
            else:
                inventory[key] = {
                    **row,
                    "active": previous["active"],
                    "entry_event_key": previous["entry_event_key"],
                    "change_event_keys": previous["change_event_keys"]
                    + [row["event_key"]],
                }
    return list(inventory.values()), issues


def closing_summary(page):
    """Read the closing table's three numeric columns at their printed positions."""
    text = page.get_text()
    if "1 + 2 - 3" not in text and "1+2-3" not in text:
        return None
    glyphs = []
    for span in page.get_texttrace():
        for group in glyph_groups(span["chars"]):
            glyphs.append(
                {
                    **group,
                    "decoded": group["text"],
                    "font": span["font"],
                    "ordinal": len(glyphs),
                }
            )
    numeric_rows = []
    for line in text_lines(glyphs):
        values = [
            w
            for w in line
            if w["x"] > page.rect.width * 0.67 and re.fullmatch(r"\d+", w["candidate"])
        ]
        if len(values) == 3:
            nums = [int(w["candidate"]) for w in values]
            numeric_rows.append(
                {
                    "y": values[0]["y"],
                    "male": nums[0],
                    "female": nums[1],
                    "total": nums[2],
                }
            )
    interpreted = interpret_summary(numeric_rows)
    return {
        "page": page.number + 1,
        "numeric_rows": numeric_rows,
        "raw_text": text,
        **interpreted,
    }


def interpret_summary(rows):
    """Validate the observed eight-row closing-table layout without repairing totals."""
    if len(rows) != 8:
        return {"format_supported": False, "arithmetic_valid": False}
    (
        base,
        additions,
        gross,
        deletions,
        deleted_total,
        final,
        corrections,
        corrected_total,
    ) = rows
    checks = [r["male"] + r["female"] == r["total"] for r in rows]
    for key in ("male", "female", "total"):
        checks.extend(
            [
                base[key] + additions[key] == gross[key],
                deletions[key] == deleted_total[key],
                gross[key] - deleted_total[key] == final[key],
                corrections[key] == corrected_total[key],
            ]
        )
    return {
        "format_supported": True,
        "arithmetic_valid": all(checks),
        "base_total": base["total"],
        "addition_total": additions["total"],
        "deletion_total": deleted_total["total"],
        "correction_total": corrected_total["total"],
        "final_total": final["total"],
    }


def npr_heading(raw):
    """Require the explicit assembly-voting restriction, not NPR in a locality."""
    return bool(re.search(r"\(\s*एन\s+पी\s+आर\s*-\s*[^)]*मतदान[^)]*\)", raw))


def component_number(raw, mangal, references):
    """Read a component heading, verifying its number and title if damaged."""
    match = re.search(r"घटक\.?\s*([123])", raw)
    if match:
        return match[1]
    if "घ" not in raw or "सूची" not in raw:
        return None
    titles = {"1": "परिवर्धन", "2": "विलोपन", "3": "संशोधन"}
    found = set()
    for line in text_lines(mangal, references):
        for index in range(len(line) - 3):
            prefix, number, title, suffix = line[index : index + 4]
            if (
                re.fullmatch(r"घ[^\s]{0,2}क\.?", prefix["candidate"])
                and number["candidate"] in titles
                and title["candidate"] == titles[number["candidate"]]
                and suffix["candidate"] == "सूची"
                and all(word["verified"] for word in (number, title, suffix))
            ):
                found.add(number["candidate"])
    return next(iter(found)) if len(found) == 1 else None


def page_sequence(pages, physical_count):
    """Check printed page numbers without treating absent footers as proof."""
    numbered = [p for p in pages if p["printed_page"] is not None]
    totals = sorted({p["printed_page_total"] for p in numbered})
    if not numbered:
        return {"valid": None, "physical_pages": physical_count, "printed_totals": []}
    disagreements = [
        {"physical": p["page"], "printed": p["printed_page"]}
        for p in numbered
        if p["printed_page"] != p["page"]
    ]
    valid = len(totals) == 1 and totals[0] == physical_count and not disagreements
    if valid and len(numbered) != len(pages):
        valid = None
    return {
        "valid": valid,
        "physical_pages": physical_count,
        "printed_totals": totals,
        "numbered_pages_checked": len(numbered),
        "position_disagreements": disagreements,
    }


def parse_pdf(path, references):
    import pymupdf

    events, pages, summaries = [], [], []
    kind, eligible, supplement = "base", True, None
    with pymupdf.open(path) as document:
        fonts = FontOutlines(document)
        physical_count = len(document)
        for page in document:
            if page.number < 2:
                continue
            raw = page.get_text()
            header = raw
            mangal, _ = extract_page(page, fonts.page(page), references)
            component = component_number(header, mangal, references)
            if component:
                kind = {"1": "addition", "2": "deletion", "3": "correction"}[component]
                eligible = not npr_heading(header)
            elif kind == "base" and npr_heading(header):
                eligible = False
            supplementary = re.search(r"अनुपूरक सूची संख्याः\s*(\d+)", header)
            if supplementary:
                supplement = supplementary[1]
            summary = closing_summary(page)
            if summary:
                summaries.append(summary)
            other = []
            for span in page.get_texttrace():
                if "Mangal" in span["font"]:
                    continue
                for group in glyph_groups(span["chars"]):
                    other.append(
                        {
                            **group,
                            "decoded": group["text"],
                            "font": span["font"],
                            "ordinal": len(other),
                        }
                    )
            rectangles = [
                tuple(item[1])
                for drawing in page.get_drawings()
                for item in drawing["items"]
                if item[0] == "re"
            ]
            frames = card_frames(rectangles)
            rows = []
            for box in frames:
                row = parse_card(box, mangal, other, references)
                if row:
                    row.update(
                        {
                            "filename": path.name,
                            "page": page.number + 1,
                            "event_key": (
                                f"{path.name}:{page.number + 1}:{len(rows) + 1}"
                            ),
                            "event_type": kind,
                            "assembly_eligible": eligible,
                            "supplement": supplement,
                        }
                    )
                    rows.append(row)
            events.extend(rows)
            footer = page.get_text(
                clip=pymupdf.Rect(
                    page.rect.width * 0.75,
                    page.rect.height - 55,
                    page.rect.width,
                    page.rect.height,
                ),
                sort=True,
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
                    "source_name_labels": raw.count("मतदाता"),
                    "events": len(rows),
                    "event_type": kind,
                    "assembly_eligible": eligible,
                }
            )
    inventory, issues = reconcile(events)
    sequence = page_sequence(pages, physical_count)
    if sequence["valid"] is False:
        issues.append({"reason": "source_page_sequence_mismatch", **sequence})
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in events + inventory:
        row["source_sha256"] = source_hash
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
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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
            "accepted_names": sum(
                r["elector_name"] is not None for r in inventory if r["active"]
            ),
            "active_total": sum(r["active"] for r in inventory),
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


FIELD_DESCRIPTIONS = {
    "filename": "Source PDF basename; interpreted with its SHA-256.",
    "source_sha256": "SHA-256 of the exact source PDF bytes.",
    "event_key": "PDF basename, one-based page and accepted card ordinal.",
    "event_type": (
        "base, addition, deletion or correction, from source section headers."
    ),
    "supplement": "Printed supplement number; null before a supplement header.",
    "number": (
        "Elector serial with validated grouping commas removed; otherwise unchanged."
    ),
    "number_raw": "Printed serial before grouping-comma normalization.",
    "id": "EPIC-like identifier recovered from the card header; null if ambiguous.",
    "elector_name": ("Name passing rendering verification with no source placeholder."),
    "name_candidate": "Decoded name candidate, retained even when verification fails.",
    "name_issue": "missing_name, source_placeholder, unverified_rendering, or null.",
    "relative_issue": "Relative-name missing/placeholder/rendering issue, or null.",
    "relative_name": (
        "Relative name accepted only when its line passes rendering verification."
    ),
    "relative_candidate": "Decoded relative-name candidate, including unverified text.",
    "relative_type": "Literal Hindi relationship label; no inferred relationship.",
    "house_no": "House field candidate kept as text, never an elector serial.",
    "age": "Printed age digits kept as text; null when no labeled value parses.",
    "sex_candidate": (
        "Decoded source sex text without categorical correction or inference."
    ),
    "header_raw": "Position-ordered source text from the card header.",
    "raw_cell": "Position-ordered source Mangal text inside the card.",
    "candidate_cell": "Decoded card text; unverified candidates remain visible.",
    "deletion_stamp_raw": (
        "Position-ordered bold Latin card glyphs used to read a deletion stamp."
    ),
    "entry_event_key": (
        "Inventory only: original base/addition event establishing the identity."
    ),
    "page": "One-based source PDF page.",
    "bbox": "Card bounds [x0, y0, x1, y1] in PDF points, top-left origin.",
    "assembly_eligible": (
        "False only for source sections labeled NPR/ineligible for assembly voting."
    ),
    "deleted_stamp": "Whether the literal DELETED stamp was recovered from the card.",
    "active": (
        "Inventory only: retained after explicit deletion evidence; null "
        "in event ledger."
    ),
    "change_event_keys": (
        "Inventory only: chronological applied correction/deletion event keys."
    ),
}


def output_schema():
    import pyarrow as pa

    strings = [
        "filename",
        "source_sha256",
        "event_key",
        "event_type",
        "supplement",
        "number",
        "number_raw",
        "id",
        "elector_name",
        "name_candidate",
        "name_issue",
        "relative_name",
        "relative_candidate",
        "relative_issue",
        "relative_type",
        "house_no",
        "age",
        "sex_candidate",
        "header_raw",
        "raw_cell",
        "candidate_cell",
        "deletion_stamp_raw",
        "entry_event_key",
    ]
    return pa.schema(
        [(name, pa.string()) for name in strings]
        + [
            ("page", pa.int32()),
            ("bbox", pa.list_(pa.float64())),
            ("assembly_eligible", pa.bool_()),
            ("deleted_stamp", pa.bool_()),
            ("active", pa.bool_()),
            ("change_event_keys", pa.list_(pa.string())),
        ]
    )


def validate_inputs(split, roots):
    names = [name for values in split.values() for name in values]
    if not names or len(set(names)) != len(names):
        raise ValueError("Split must be nonempty with unique PDF filenames")
    paths, hashes = {}, set()
    for name in names:
        found = [root / name for root in roots if (root / name).is_file()]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one PDF for {name}")
        digest = hashlib.sha256(found[0].read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate PDF content: {name}")
        hashes.add(digest)
        paths[name] = found[0]
    return paths


_WORKER_REFERENCES = None


def initialize_worker(font_paths):
    global _WORKER_REFERENCES
    _WORKER_REFERENCES = ReferenceFonts(font_paths)


def parse_job(path):
    if _WORKER_REFERENCES is None:
        raise RuntimeError("Worker fonts were not initialized")
    try:
        events, inventory, audit = parse_pdf(path, _WORKER_REFERENCES)
        audit["parse_status"] = "parsed" if events else "no_events"
        return events, inventory, audit
    except Exception as error:
        audit = {
            "filename": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parse_status": "error",
            "event_counts": {},
            "events": 0,
            "inventory": 0,
            "active_total": 0,
            "active_assembly": 0,
            "active_npr": 0,
            "accepted_names": 0,
            "fully_reconciled": False,
            "printed_final_total": None,
            "printed_arithmetic_valid": False,
            "count_differences": {},
            "pages": [],
            "closing_summaries": [],
            "issues": [
                {
                    "reason": "parse_error",
                    "type": type(error).__name__,
                    "message": str(error),
                }
            ],
        }
        return [], [], audit
    finally:
        _WORKER_REFERENCES._shape_cache.clear()


def main():
    import time
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import ExitStack
    from importlib.metadata import version

    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, action="append", required=True)
    parser.add_argument("--reference-font", type=Path, action="append", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    split = json.loads(args.split.read_text())
    paths = validate_inputs(split, args.pdf_dir)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    references = ReferenceFonts(args.reference_font)
    parts, schema = [], output_schema()
    jobs = [(label, name) for label, filenames in split.items() for name in filenames]
    with ExitStack() as stack:
        event_out = stack.enter_context(
            pq.ParquetWriter(
                args.out_dir / "events.parquet", schema, compression="zstd"
            )
        )
        inventory_out = stack.enter_context(
            pq.ParquetWriter(
                args.out_dir / "inventory.parquet", schema, compression="zstd"
            )
        )
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
        for (label, _), (events, inventory, audit) in zip(jobs, results):
            audit["split"] = label
            parts.append(audit)
            for rows, output in [(events, event_out), (inventory, inventory_out)]:
                output.write_table(pa.Table.from_pylist(rows, schema=schema))
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in audit.items()
                        if k not in ("pages", "closing_summaries", "issues")
                    }
                ),
                flush=True,
            )
    summary = {}
    for label in split:
        selected = [part for part in parts if part["split"] == label]
        summary[label] = {
            "pdfs": len(selected),
            **{
                key: sum(part[key] for part in selected)
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
            "parts_with_issues": sum(bool(part["issues"]) for part in selected),
            "failed_parts": sum(part["parse_status"] == "error" for part in selected),
            "parts_without_events": sum(
                part["parse_status"] == "no_events" for part in selected
            ),
            "parts_matching_final_total": sum(
                part["count_differences"].get("active") == 0 for part in selected
            ),
        }
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
            "description": FIELD_DESCRIPTIONS[field.name],
        }
        for field in schema
    ]
    (args.out_dir / "SCHEMA.json").write_text(
        json.dumps({"fields": fields}, indent=2) + "\n"
    )
    audit = {
        "summary": summary,
        "roll_year": 2018,
        "geography": "Historical Jammu and Kashmir, including present-day Ladakh",
        "parts": parts,
        "reference_fonts": references.sources,
        "script_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("jk_hindi_rows.py", "hindi_text.py", "hindi_glyphs.py")
        },
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "artifacts": {
            name: hashlib.sha256((args.out_dir / name).read_bytes()).hexdigest()
            for name in ("events.parquet", "inventory.parquet", "SCHEMA.json")
        },
        "versions": {
            name: version(name)
            for name in ("PyMuPDF", "fonttools", "uharfbuzz", "pyarrow")
        },
        "elapsed_seconds": time.monotonic() - started,
        "workers": args.workers,
        "limitations": [
            "Inventory keys combine serials and assembly eligibility.",
            "Corrections add no electors; duplicate deletion evidence subtracts once.",
            "Total comparisons combine assembly and NPR inventories.",
            "Rendering agreement is not a name-accuracy estimate.",
            "Issues remain explicit; surname and model artifacts are unchanged.",
        ],
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
