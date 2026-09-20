"""Read Urdu text fields whose source outlines match a supplied reference font.

The PDF's Unicode is checked against physical glyphs before NFKC normalization.
Contour order and starting points are immaterial. A one-font-unit coordinate
tolerance accommodates subpixel differences in exported composite glyphs, while
preserving every contour, curve flag and dot position.
Names with placeholders, missing labels or unverified glyphs are withheld.
Small polygonal dot variants additionally require a uniquely matching Arabic
letter, a verified body and a fixed coordinate bound of 1/512 em.
"""

import hashlib
import io
import re
import unicodedata
from collections import Counter

from model_training.prep_er_data.jk_urdu_sections import BASE14


def outline_key(font, gid):
    """Identify exact closed contours independent of contour order and start."""
    if gid <= 0 or gid >= len(font.getGlyphOrder()):
        return None
    glyph = font["glyf"][font.getGlyphName(gid)]
    if not glyph.numberOfContours:
        return None
    coordinates, ends, flags = glyph.getCoordinates(font["glyf"])
    contours, start = [], 0
    for end in ends:
        points = tuple(
            (coordinates[i][0], coordinates[i][1], flags[i])
            for i in range(start, end + 1)
        )
        contours.append(min(points[i:] + points[:i] for i in range(len(points))))
        start = end + 1
    return hashlib.sha256(
        repr((font["head"].unitsPerEm, sorted(contours))).encode()
    ).hexdigest()


def contours(font, gid):
    """Return closed point sequences without changing geometry or curve flags."""
    glyph = font["glyf"][font.getGlyphName(gid)]
    coordinates, ends, flags = glyph.getCoordinates(font["glyf"])
    result, start = [], 0
    for end in ends:
        result.append(
            tuple(
                (coordinates[i][0], coordinates[i][1], flags[i])
                for i in range(start, end + 1)
            )
        )
        start = end + 1
    return result


def contours_match(source, reference, tolerance=1):
    """Match all closed contours one-to-one within a fixed coordinate bound."""
    if len(source) != len(reference) or not source:
        return False
    options = []
    for first in source:
        candidates = []
        for j, second in enumerate(reference):
            if len(first) != len(second):
                continue
            if any(
                all(
                    a[2] == b[2]
                    and abs(a[0] - b[0]) <= tolerance
                    and abs(a[1] - b[1]) <= tolerance
                    for a, b in zip(first, second[i:] + second[:i], strict=True)
                )
                for i in range(len(second))
            ):
                candidates.append(j)
        if not candidates:
            return False
        options.append(candidates)
    assigned = {}

    def assign(i, visited):
        for j in options[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or assign(assigned[j], visited):
                assigned[j] = i
                return True
        return False

    return all(assign(i, set()) for i in range(len(source)))


def native_text(text):
    """Normalize presentation forms while rejecting non-Arabic source text."""
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    if not text or not any(c.isalpha() for c in text):
        return None
    if all(
        c in " .-'\u2019"
        or (0x0600 <= ord(c) <= 0x06FF and unicodedata.category(c)[0] in "LM")
        for c in text
    ):
        return text
    return None


def contour_topology(value):
    """Index contours by point counts and flags without dropping any marks."""
    return tuple(sorted((len(c), tuple(sorted(p[2] for p in c))) for c in value))


def is_small_dot(value, units):
    """Recognize only four-point straight contours no larger than 1/8 em."""
    return (
        len(value) == 4
        and all(p[2] == 1 for p in value)
        and all(
            0 < max(p[axis] for p in value) - min(p[axis] for p in value) <= units / 8
            for axis in (0, 1)
        )
    )


class DotVariants:
    """Corroborate small dot changes only when the Arabic meaning is unique."""

    def __init__(self, reference):
        """Index all Arabic reference meanings, retaining Unicode ambiguity."""
        self.units = reference["head"].unitsPerEm
        self.reference = {}
        self.index = {}
        self.cache = {}
        for codepoint, name in (reference.getBestCmap() or {}).items():
            if not (
                0x0600 <= codepoint <= 0x06FF
                or 0xFB50 <= codepoint <= 0xFDFF
                or 0xFE70 <= codepoint <= 0xFEFF
            ):
                continue
            gid = reference.getGlyphID(name)
            if not outline_key(reference, gid):
                continue
            if gid not in self.reference:
                value = contours(reference, gid)
                self.reference[gid] = (value, set())
                self.index.setdefault(contour_topology(value), []).append(gid)
            self.reference[gid][1].add(unicodedata.normalize("NFKC", chr(codepoint)))

    def match(self, font, gid, codepoint):
        """Keep the source spelling only if bounded geometry uniquely supports it."""
        if font["head"].unitsPerEm != self.units:
            return False
        key = outline_key(font, gid)
        if key is None:
            return False
        if key not in self.cache:
            source = contours(font, gid)
            bodies = [c for c in source if not is_small_dot(c, self.units)]
            dots = [c for c in source if is_small_dot(c, self.units)]
            meanings = set()
            anchored = False
            for candidate in self.index.get(contour_topology(source), []):
                target, texts = self.reference[candidate]
                if not contours_match(source, target, tolerance=self.units / 512):
                    continue
                meanings.update(texts)
                target_bodies = [c for c in target if not is_small_dot(c, self.units)]
                if dots and bodies and contours_match(bodies, target_bodies):
                    anchored = True
            self.cache[key] = meanings if anchored else set()
        return self.cache[key] == {unicodedata.normalize("NFKC", chr(codepoint))}


class UnicodeFonts:
    """Verify source Unicode against an external font within one font unit."""

    def __init__(self, document, reference):
        """Keep source and reference glyph identities separate."""
        self.document = document
        self.reference = reference
        self.cmap = reference.getBestCmap() or {}
        self.fonts = {}
        self.source_keys = {}
        self.reference_keys = {}
        self.matches = {}
        self.dot_matches = {}
        self.base_fonts = {}
        self.checks = Counter()
        self.dot_variants = DotVariants(reference)

    def page(self, page):
        """Return character evidence with glyph checks for one source page."""
        import pymupdf
        from fontTools.ttLib import TTFont

        by_name = {}
        base_names = set()
        for xref, extension, _, name, *_ in page.get_fonts():
            if extension == "n/a" and name in BASE14:
                base_names.add(name)
                if name not in self.base_fonts:
                    self.base_fonts[name] = pymupdf.Font(name)
            if extension != "ttf":
                continue
            name = name.split("+")[-1]
            if xref not in self.fonts:
                self.fonts[xref] = TTFont(
                    io.BytesIO(self.document.extract_font(xref)[3])
                )
            by_name.setdefault(name, []).append(xref)
        result = []
        for span in page.get_texttrace():
            for codepoint, gid, origin, bbox in span["chars"]:
                keys = set()
                for xref in by_name.get(span["font"], []):
                    key = (xref, gid)
                    if key not in self.source_keys:
                        self.source_keys[key] = outline_key(self.fonts[xref], gid)
                    if self.source_keys[key] is not None:
                        keys.add(self.source_keys[key])
                reference_name = self.cmap.get(codepoint)
                if reference_name not in self.reference_keys:
                    self.reference_keys[reference_name] = (
                        outline_key(
                            self.reference, self.reference.getGlyphID(reference_name)
                        )
                        if reference_name
                        else None
                    )
                expected = self.reference_keys[reference_name]
                verified = bool(expected and keys == {expected})
                if not verified and expected and keys:
                    matches = []
                    for xref in by_name.get(span["font"], []):
                        if self.source_keys[(xref, gid)] is None:
                            continue
                        key = (xref, gid, codepoint)
                        if key not in self.matches:
                            font = self.fonts[xref]
                            self.matches[key] = font[
                                "head"
                            ].unitsPerEm == self.reference[
                                "head"
                            ].unitsPerEm and contours_match(
                                contours(font, gid),
                                contours(
                                    self.reference,
                                    self.reference.getGlyphID(reference_name),
                                ),
                            )
                        matches.append(self.matches[key])
                    verified = bool(matches) and all(matches)
                    if not verified and len(keys) == 1:
                        variant_key = (next(iter(keys)), codepoint)
                        if variant_key not in self.dot_matches:
                            font = next(
                                self.fonts[xref]
                                for xref in by_name[span["font"]]
                                if self.source_keys[(xref, gid)] is not None
                            )
                            self.dot_matches[variant_key] = self.dot_variants.match(
                                font, gid, codepoint
                            )
                        verified = self.dot_matches[variant_key]
                        if verified:
                            self.checks["verified_dot_variant"] += 1
                if chr(codepoint).isspace():
                    verified = bool(by_name.get(span["font"])) and not keys and gid >= 0
                if span["font"] in base_names and span["font"] not in by_name:
                    verified = (
                        32 <= codepoint <= 126
                        and self.base_fonts[span["font"]].has_glyph(codepoint) == gid
                    )
                self.checks["verified" if verified else "unverified"] += 1
                result.append(
                    {
                        "char": chr(codepoint),
                        "gid": gid,
                        "origin": origin,
                        "bbox": bbox,
                        "font": span["font"],
                        "verified": verified,
                    }
                )
        return result


def contains(box, point):
    """Include a baseline origin with a small PDF coordinate tolerance."""
    return (
        box[0] - 0.01 <= point[0] <= box[2] + 0.01
        and box[1] - 0.01 <= point[1] <= box[3] + 0.01
    )


def verified_span(span, glyphs):
    """Require complete character coverage as well as matching physical outlines."""
    expected = Counter(c for c in span["text"] if not c.isspace())
    source = [
        g
        for g in glyphs
        if g["font"] == span["font"]
        and contains(span["bbox"], g["origin"])
        and not g["char"].isspace()
    ]
    return (
        bool(expected)
        and expected == Counter(g["char"] for g in source)
        and all(g["verified"] for g in source)
    )


def compact(text):
    """Compare fixed field labels after Unicode presentation normalization."""
    return "".join(unicodedata.normalize("NFKC", text).split())


RELATIONSHIPS = {"باپ", "خاوند", "ماں", "والد", "والدہ", "پسرپروردہ", "دیگر"}


def field_value(label, end_y, spans, glyphs):
    """Read text to the left of a label and its following continuation lines."""
    baseline = label["origin"][1]
    candidates = [
        s
        for s in spans
        if baseline - 0.5 <= s["origin"][1] < end_y - 0.5
        and s is not label
        and (s["origin"][1] > baseline + 1 or s["bbox"][2] <= label["bbox"][0] + 0.1)
        and s["text"].strip() not in {":", ""}
    ]
    candidates.sort(key=lambda s: (round(s["origin"][1], 1), -s["bbox"][2]))
    raw = " ".join(s["text"].strip() for s in candidates) or None
    normalized = native_text(raw) if raw else None
    if normalized is None:
        return raw, None, "unsupported_name_text" if raw else "missing_name"
    if len(candidates) != 1:
        return raw, None, "split_name_span_unverified"
    if not verified_span(label, glyphs):
        return raw, None, "unverified_field_label"
    if not all(verified_span(s, glyphs) for s in candidates):
        return raw, None, "unverified_unicode_outline"
    return raw, normalized, None


def numeric_field(label, controls, spans, glyphs, *, house=False):
    """Read a value in its printed control column, excluding nearby controls."""
    left = max(
        [s["bbox"][2] for s in controls if s["bbox"][2] < label["bbox"][0]] or [0]
    )
    candidates = []
    for span in spans:
        text = unicodedata.normalize("NFKC", span["text"]).strip()
        text = "".join(
            str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text
        )
        pattern = r"[A-Za-z0-9][A-Za-z0-9/.-]*" if house else r"[0-9]{1,3}"
        if (
            label["origin"][1] - 0.5 <= span["origin"][1] <= label["origin"][1] + 12
            and left < span["bbox"][0] < span["bbox"][2] <= label["bbox"][0] + 0.1
            and re.fullmatch(pattern, text)
            and any(c.isdecimal() for c in text)
            and verified_span(span, glyphs)
        ):
            candidates.append(text)
    return candidates[0] if len(candidates) == 1 else None


def parse_card(box, page_spans, glyphs):
    """Extract native fields without changing the existing identity ledger."""
    spans = [s for s in page_spans if contains(box, s["origin"])]
    glyphs = [g for g in glyphs if contains(box, g["origin"])]
    labels = [s for s in spans if compact(s["text"]) == "نامووٹر"]
    relations = [s for s in spans if compact(s["text"]) in RELATIONSHIPS]
    controls = [
        s
        for s in spans
        if compact(s["text"]) in {"خانہ", "خانہنمبر", "عمر"}
        or compact(s["text"]).startswith("جنس")
    ]
    result = dict.fromkeys(
        [
            "elector_name",
            "name_candidate",
            "relative_name",
            "relative_candidate",
            "relative_type",
            "house_no",
            "age",
            "sex_candidate",
        ]
    )
    result.update(
        name_issue="missing_or_ambiguous_name_label",
        relative_issue="missing_or_ambiguous_relative_label",
    )
    if len(labels) == 1:
        end = min(
            [
                s["origin"][1]
                for s in relations + controls
                if s["origin"][1] > labels[0]["origin"][1]
            ]
            or [box[3]]
        )
        result["name_candidate"], result["elector_name"], result["name_issue"] = (
            field_value(labels[0], end, spans, glyphs)
        )
    if len(relations) == 1:
        relation = relations[0]
        end = min(
            [s["origin"][1] for s in controls if s["origin"][1] > relation["origin"][1]]
            or [box[3]]
        )
        result["relative_type"] = unicodedata.normalize(
            "NFKC", relation["text"]
        ).strip()
        (
            result["relative_candidate"],
            result["relative_name"],
            result["relative_issue"],
        ) = field_value(relation, end, spans, glyphs)
    for prefix, field in [("خانہ", "house_no"), ("عمر", "age")]:
        labels = [s for s in controls if compact(s["text"]).startswith(prefix)]
        if len(labels) == 1 and verified_span(labels[0], glyphs):
            result[field] = numeric_field(
                labels[0], controls, spans, glyphs, house=field == "house_no"
            )
    sexes = [s for s in controls if compact(s["text"]).startswith("جنس")]
    if len(sexes) == 1 and verified_span(sexes[0], glyphs):
        text = compact(sexes[0]["text"])[len("جنس") :]
        if text in {"مرد", "عورت"}:
            result["sex_candidate"] = text
        elif not text:
            y = sexes[0]["origin"][1]
            values = [
                compact(s["text"])
                for s in spans
                if abs(s["origin"][1] - y) <= 0.5
                and s["bbox"][2] <= sexes[0]["bbox"][0] + 0.1
                and compact(s["text"]) in {"مرد", "عورت"}
                and verified_span(s, glyphs)
            ]
            result["sex_candidate"] = values[0] if len(values) == 1 else None
    return result


def recover_part(payload, pdf_dir, reference_path):
    """Enrich current inventory cards while retaining every source identity."""
    import pymupdf
    from fontTools.ttLib import TTFont

    from model_training.prep_er_data.jk_urdu_rows import file_sha256

    rows, part = payload
    source = pdf_dir / part["filename"]
    if file_sha256(source) != part["source_sha256"]:
        raise ValueError("Source PDF hash changed")
    if any(
        r["filename"] != part["filename"] or r["source_sha256"] != part["source_sha256"]
        for r in rows
    ):
        raise ValueError("Inventory identity does not match its source PDF")
    by_page = {}
    for row in rows:
        by_page.setdefault(row["page"], []).append(row)
    updated = {}
    with pymupdf.open(source) as document:
        fonts = UnicodeFonts(document, TTFont(reference_path))
        for number, page_rows in by_page.items():
            page = document[number - 1]
            glyphs = fonts.page(page)
            spans = [
                s
                for b in page.get_text("dict")["blocks"]
                for line in b.get("lines", [])
                for s in line["spans"]
            ]
            for row in page_rows:
                fields = parse_card(row["bbox"], spans, glyphs)
                fields["raw_cell"] = "\n".join(
                    s["text"] for s in spans if contains(row["bbox"], s["origin"])
                )
                updated[row["event_key"]] = {**row, **fields}
    if len(updated) != len(rows):
        raise ValueError("Inventory repeats a current event key")
    output = [updated[row["event_key"]] for row in rows]
    names = {
        "accepted_names": sum(r["elector_name"] is not None for r in output),
        "accepted_active_names": sum(
            r["active"] and r["elector_name"] is not None for r in output
        ),
        "accepted_active_relatives": sum(
            r["active"] and r["relative_name"] is not None for r in output
        ),
        "active_house_numbers": sum(
            r["active"] and r["house_no"] is not None for r in output
        ),
        "name_issue_counts": dict(
            Counter(r["name_issue"] or "verified" for r in output)
        ),
        "relative_issue_counts": dict(
            Counter(r["relative_issue"] or "verified" for r in output)
        ),
        "glyph_checks": dict(fonts.checks),
    }
    return output, {**part, **part["counts"], "native_text": names}


def main():
    """Produce a native inventory from a fixed reconciled source artifact."""
    import argparse
    import itertools
    import json
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from importlib.metadata import version
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from model_training.prep_er_data.jk_urdu_rows import emit, file_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--reference-font", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    audit = json.loads(args.source_audit.read_text())
    if file_sha256(args.inventory) != audit["artifacts"]["inventory.parquet"]:
        raise ValueError("Input inventory hash changed")
    reference_sha = file_sha256(args.reference_font)
    parts = {p["filename"]: p for p in audit["parts"]}
    if len(parts) != len(audit["parts"]):
        raise ValueError("Source audit repeats a filename")
    parquet = pq.ParquetFile(args.inventory)
    records = (
        row
        for batch in parquet.iter_batches(batch_size=10000)
        for row in batch.to_pylist()
    )
    payloads = (
        (list(rows), parts[name])
        for name, rows in itertools.groupby(records, key=lambda r: r["filename"])
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    results, seen = [], set()
    worker = partial(
        recover_part, pdf_dir=args.pdf_dir, reference_path=args.reference_font
    )
    with (
        pq.ParquetWriter(
            args.out_dir / "inventory.parquet", parquet.schema_arrow, compression="zstd"
        ) as writer,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        for batch in itertools.batched(payloads, args.workers * 2):
            for rows, part in pool.map(worker, batch):
                if part["filename"] in seen:
                    raise ValueError("Inventory rows are not contiguous by source PDF")
                seen.add(part["filename"])
                if len(rows) != part["inventory"]:
                    raise ValueError("Native extraction changed an inventory count")
                writer.write_table(
                    pa.Table.from_pylist(rows, schema=parquet.schema_arrow)
                )
                results.append(part)
                emit({"filename": part["filename"], **part["native_text"]})
    if seen != parts.keys():
        raise ValueError("Input audit and inventory source coverage disagree")
    summary = {
        **audit["summary"],
        **{
            key: sum(p["native_text"][key] for p in results)
            for key in [
                "accepted_names",
                "accepted_active_names",
                "accepted_active_relatives",
                "active_house_numbers",
            ]
        },
    }
    summary["accepted_urdu_names"] = summary["accepted_names"]
    metadata = {
        "summary": summary,
        "parts": results,
        "source_inventory_sha256": file_sha256(args.inventory),
        "source_audit_sha256": file_sha256(args.source_audit),
        "reference_font": {
            "filename": args.reference_font.name,
            "sha256": reference_sha,
        },
        "outline_coordinate_tolerance": {
            "maximum_font_units": 1,
            "contours_and_curve_flags_preserved": True,
            "font_units_must_match": True,
            "dot_variant_fallback": {
                "maximum_em": 1 / 512,
                "dot_shape": "Four on-curve points, width and height at most 1/8 em",
                "body_maximum_font_units": 1,
                "requires_body_and_dots": True,
                "requires_unique_arabic_nfkc_meaning": True,
                "source_unicode_is_not_replaced": True,
            },
        },
        "script_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ["jk_urdu_unicode.py", "jk_urdu_sections.py"]
        },
        "versions": {
            name: version(name) for name in ["PyMuPDF", "fonttools", "pyarrow"]
        },
        "artifacts": {
            "inventory.parquet": file_sha256(args.out_dir / "inventory.parquet")
        },
        "limitations": [
            "Native transcription checks do not establish surname identity.",
            "Split name spans and unsupported glyphs are withheld.",
            "Latin transliteration has not been performed.",
            "Original counts, source discrepancies and activity flags are preserved.",
        ],
    }
    (args.out_dir / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out_dir / "SCHEMA.json").write_text(
        json.dumps(
            {
                "fields": [
                    {"name": f.name, "type": str(f.type), "nullable": f.nullable}
                    for f in parquet.schema_arrow
                ],
                "names": (
                    "NFKC-normalized Urdu values with glyph verification; "
                    "original candidate strings retained."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    emit(summary)


if __name__ == "__main__":
    main()
