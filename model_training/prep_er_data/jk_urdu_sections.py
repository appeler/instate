"""Recognize Urdu supplement headings from verified source glyph outlines.

The catalogue contains complete heading signatures from visually checked source
pages. Unseen headings remain unclassified. Glyph numbers and damaged Unicode
mappings never establish a heading's meaning.
"""

import hashlib
import io
import json

from model_training.prep_er_data.hindi_glyphs import glyph_fingerprint

BASE14 = {
    "Helvetica",
    "Helvetica-Bold",
    "Helvetica-Oblique",
    "Helvetica-BoldOblique",
    "Times-Roman",
    "Times-Bold",
    "Times-Italic",
    "Times-BoldItalic",
    "Courier",
    "Courier-Bold",
    "Courier-Oblique",
    "Courier-BoldOblique",
    "Symbol",
    "ZapfDingbats",
}


def cff_fingerprint(font, gid):
    """Hash CFF drawing commands and the font's coordinate transform."""
    from fontTools.pens.recordingPen import RecordingPen

    if gid <= 0 or gid >= len(font.charset):
        return None
    pen = RecordingPen()
    font.CharStrings[font.charset[gid]].draw(pen)
    if not pen.value:
        return None
    return hashlib.sha256(
        json.dumps(["cff", font.FontMatrix, pen.value]).encode()
    ).hexdigest()


class HeaderFonts:
    """Resolve source outlines within a document, preserving font ambiguity."""

    def __init__(self, document):
        """Cache embedded fonts and the outlines requested by source headers."""
        self.document = document
        self.fonts = {}
        self.outlines = {}

    def page(self, page, spans):
        """Return unambiguous nonempty outlines for the requested header glyphs."""
        import pymupdf
        from fontTools.cffLib import CFFFontSet
        from fontTools.ttLib import TTFont

        wanted = {}
        for span in spans:
            wanted.setdefault(span["font"], set()).update(c[1] for c in span["chars"])
        candidates = {}
        for xref, extension, _, name, *_ in page.get_fonts():
            name = name.split("+")[-1]
            base14 = extension == "n/a" and name in BASE14
            if name not in wanted or (extension not in {"ttf", "cff"} and not base14):
                continue
            if xref not in self.fonts:
                data = (
                    pymupdf.Font(name).buffer
                    if base14
                    else self.document.extract_font(xref)[3]
                )
                if extension == "ttf":
                    self.fonts[xref] = TTFont(io.BytesIO(data))
                else:
                    fonts = CFFFontSet()
                    fonts.decompile(io.BytesIO(data), TTFont())
                    if len(fonts) != 1:
                        raise ValueError("Ambiguous embedded CFF font collection")
                    self.fonts[xref] = fonts[0]
                self.outlines[xref] = {}
            for gid in wanted[name]:
                if gid not in self.outlines[xref]:
                    fingerprint = (
                        glyph_fingerprint if extension == "ttf" else cff_fingerprint
                    )
                    self.outlines[xref][gid] = fingerprint(self.fonts[xref], gid)
                outline = self.outlines[xref][gid]
                if outline is not None:
                    candidates.setdefault((name, gid), set()).add(outline)
        return {
            key: next(iter(values)) if len(values) == 1 else None
            for key, values in candidates.items()
        }


def heading_markers(spans, outlines, catalog):
    """Require the outlined section word, hyphen and neighboring numeral."""
    glyphs = [
        {
            "font": span["font"],
            "outline": outlines.get((span["font"], c[1])),
            "x": c[2],
            "y": c[3],
        }
        for span in spans
        for c in span["chars"]
    ]
    glyphs = [g for g in glyphs if g["outline"] is not None]
    markers = []
    words = [catalog["section_word"], *catalog.get("additional_section_words", [])]
    for first in glyphs:
        matching_words = [word for word in words if first["outline"] == word[0]]
        if not matching_words:
            continue
        word_found = False
        for expected in matching_words:
            remaining = [
                [
                    g
                    for g in glyphs
                    if g["font"] == first["font"]
                    and g["outline"] == outline
                    and 0 < g["x"] - first["x"] < 20
                    and abs(g["y"] - first["y"]) < 8
                ]
                for outline in expected[1:]
            ]
            if (
                all(len(options) == 1 for options in remaining)
                and remaining[0][0]["x"] < remaining[1][0]["x"]
            ):
                word_found = True
        if not word_found:
            continue
        nearby = [
            g
            for g in glyphs
            if 0 < first["x"] - g["x"] < 30 and abs(first["y"] - g["y"]) < 1
        ]
        for digit in nearby:
            if digit["font"] != first["font"]:
                continue
            number = catalog["digits"].get(digit["outline"])
            if number is None:
                continue
            if not any(
                g["outline"]
                in [catalog["hyphen"], *catalog.get("additional_hyphens", [])]
                and 0 < g["x"] - digit["x"] < 12
                for g in nearby
            ):
                continue
            line = [
                g
                for g in glyphs
                if g["font"] == first["font"]
                and first["y"] - 20 < g["y"] < first["y"] + 8
            ]
            signature = [
                g["outline"] for g in sorted(line, key=lambda g: (g["x"], g["y"]))
            ]
            digest = hashlib.sha256("|".join(signature).encode()).hexdigest()
            label = catalog["headings"].get(digest)
            markers.append(
                {
                    "component": number,
                    "y": first["y"],
                    "signature": digest,
                    "label": label,
                }
            )
    verified_baselines = [m["y"] for m in markers if m["label"] is not None]
    for glyph in glyphs:
        x, y = glyph["x"], glyph["y"]
        if catalog["digits"].get(glyph["outline"]) not in (1, 2, 3) or x < 480:
            continue
        if verified_baselines and y < min(verified_baselines):
            continue
        if any(abs(marker["y"] - y) < 1 for marker in markers):
            continue
        if any(
            other["font"] == glyph["font"]
            and other["outline"]
            in [catalog["hyphen"], *catalog.get("additional_hyphens", [])]
            and 0 < other["x"] - x < 12
            and abs(other["y"] - y) < 1
            for other in glyphs
        ):
            markers.append(
                {
                    "component": None,
                    "y": y,
                    "signature": None,
                    "label": None,
                    "issue": "outlined_number_without_verified_section_word",
                }
            )
    return markers


def assembly_restriction(spans, outlines, catalog):
    """Match every restriction-phrase outline at consistent source positions."""
    glyphs = [
        (outlines.get((span["font"], c[1])), c[2], c[3])
        for span in spans
        for c in span["chars"]
    ]
    for pattern in catalog.get("assembly_restriction_patterns", []):
        expected = pattern["glyphs"]
        first, last = expected[0], expected[-1]
        for start_index, start in enumerate(glyphs):
            if start[0] != first[0]:
                continue
            for end in glyphs:
                if end[0] != last[0] or end[1] <= start[1]:
                    continue
                scale = (end[1] - start[1]) / (last[1] - first[1])
                if not 0.5 <= scale <= 2:
                    continue
                used = {start_index}
                for outline, x, y in expected[1:]:
                    target_x = start[1] + scale * (x - first[1])
                    target_y = start[2] + scale * (y - first[2])
                    matches = [
                        i
                        for i, glyph in enumerate(glyphs)
                        if i not in used
                        and glyph[0] == outline
                        and abs(glyph[1] - target_x) <= 0.5
                        and abs(glyph[2] - target_y) <= 0.5
                    ]
                    if len(matches) != 1:
                        break
                    used.add(matches[0])
                else:
                    return True
    return False


def classify_pages(pages, outlines_by_page, catalog):
    """Carry only recognized section labels forward through continuation pages."""
    kind, eligible = "base", True
    result = []
    for page in pages:
        markers = heading_markers(
            page["header_spans"], outlines_by_page[page["page"]], catalog
        )
        issue = None
        if len(markers) > 1:
            kind, eligible, issue = None, None, "multiple_component_headings"
        elif markers:
            label = markers[0]["label"]
            if label is None:
                kind, eligible, issue = None, None, "unrecognized_component_heading"
            else:
                kind, eligible = label["event_type"], label["assembly_eligible"]
        restriction = assembly_restriction(
            page["header_spans"], outlines_by_page[page["page"]], catalog
        )
        if restriction and kind is not None:
            eligible = False
        result.append(
            {
                "page": page["page"],
                "event_type": kind,
                "assembly_eligible": eligible,
                "markers": markers,
                "issue": issue,
                "explicit_assembly_restriction": restriction,
            }
        )
    return result
