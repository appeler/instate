"""Recover missing Mangal mappings from identical source glyph outlines.

This is a partial text-layer repair. It does not infer unmapped conjuncts,
reorder Devanagari, parse elector rows, or transliterate names.
"""

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


def usable_unicode(text):
    return bool(text) and all(
        32 <= ord(char) <= 126 or "\u0900" <= char <= "\u097f" for char in text
    )


def glyph_groups(chars):
    """Attach virtual Unicode continuations to their physical source glyph."""
    groups = []
    for codepoint, gid, origin, bbox in chars:
        if gid == -1 and groups:
            groups[-1]["text"] += chr(codepoint)
        else:
            groups.append(
                {"gid": gid, "text": chr(codepoint), "origin": origin, "bbox": bbox}
            )
    return groups


def glyph_fingerprint(font, gid):
    """Hash exact decomposed outlines and font units; never match .notdef."""
    if gid <= 0 or gid >= len(font.getGlyphOrder()):
        return None
    glyph = font["glyf"][font.getGlyphName(gid)]
    if not glyph.numberOfContours:
        return None
    coordinates, ends, flags = glyph.getCoordinates(font["glyf"])
    payload = [font["head"].unitsPerEm, list(coordinates), list(ends), list(flags)]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


class FontOutlines:
    """Resolve glyph identities within each PDF, rejecting ambiguous subsets."""

    def __init__(self, document):
        self.document = document
        self.fonts = {}
        self.fingerprints = {}

    def page(self, page):
        from fontTools.ttLib import TTFont

        outlines = defaultdict(lambda: defaultdict(set))
        for xref, _, _, base, _, _ in page.get_fonts():
            if "Mangal" not in base:
                continue
            if xref not in self.fonts:
                _, extension, _, data = self.document.extract_font(xref)
                if extension != "ttf":
                    raise ValueError("Mangal font is not embedded TrueType")
                font = TTFont(io.BytesIO(data))
                self.fonts[xref] = font
                self.fingerprints[xref] = {
                    gid: glyph_fingerprint(font, gid)
                    for gid in range(len(font.getGlyphOrder()))
                }
            for gid, fingerprint in self.fingerprints[xref].items():
                if fingerprint is not None:
                    outlines[base.split("+")[-1]][gid].add(fingerprint)
        return {
            name: {
                gid: next(iter(values)) if len(values) == 1 else None
                for gid, values in glyphs.items()
            }
            for name, glyphs in outlines.items()
        }


def finalize_catalog(candidates):
    """Keep conflicting source assignments out of the recovery map."""
    mapping, conflicts = {}, {}
    for fingerprint, texts in candidates.items():
        if len(texts) == 1:
            mapping[fingerprint] = next(iter(texts))
        else:
            conflicts[fingerprint] = sorted(texts)
    return mapping, conflicts


def build_catalog(paths):
    import pymupdf

    candidates = defaultdict(set)
    evidence = defaultdict(set)
    sources = {}
    for path in paths:
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with pymupdf.open(path) as document:
            fonts = FontOutlines(document)
            for page in document:
                outlines = fonts.page(page)
                for span in page.get_texttrace():
                    mapping = outlines.get(span["font"], {})
                    for group in glyph_groups(span["chars"]):
                        fingerprint = mapping.get(group["gid"])
                        if fingerprint and usable_unicode(group["text"]):
                            candidates[fingerprint].add(group["text"])
                            evidence[fingerprint].add(path.name)
    mapping, conflicts = finalize_catalog(candidates)
    return {
        "revision": "mangal-outline-catalog-v1",
        "sources": sources,
        "mapping": mapping,
        "conflicts": conflicts,
        "evidence_parts": {key: sorted(value) for key, value in evidence.items()},
    }


def recover_group(group, fingerprint, catalog):
    """Preserve valid source text and expose every unresolved physical glyph."""
    original = group["text"]
    if usable_unicode(original):
        return original, "source_mapping"
    if fingerprint in catalog["mapping"]:
        return catalog["mapping"][fingerprint], "matched_outline"
    return "\ufffd", "unresolved"


def evaluate_pdf(path, catalog, output):
    import pymupdf

    statistics = Counter()
    with pymupdf.open(path) as document:
        fonts = FontOutlines(document)
        for page in document:
            outlines = fonts.page(page)
            for ordinal, span in enumerate(page.get_texttrace()):
                if "Mangal" not in span["font"]:
                    continue
                mapping = outlines.get(span["font"], {})
                groups = glyph_groups(span["chars"])
                repaired = []
                for group in groups:
                    text, status = recover_group(
                        group, mapping.get(group["gid"]), catalog
                    )
                    statistics[status] += 1
                    repaired.append(text)
                output.write(
                    json.dumps(
                        {
                            "filename": path.name,
                            "page": page.number + 1,
                            "span": ordinal,
                            "font": span["font"],
                            "raw_text": "".join(group["text"] for group in groups),
                            "decoded_visual_order": "".join(repaired),
                            "glyphs": groups,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return {
        "filename": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **statistics,
    }


def main():
    import gzip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    development = set(split["development"])
    holdout = set(split["holdout"])
    if development & holdout or not development or not holdout:
        parser.error("development and holdout must be nonempty and disjoint")
    paths = {name: args.pdf_dir / name for name in development | holdout}
    for path in paths.values():
        if not path.is_file():
            parser.error(f"source PDF is missing: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    catalog = build_catalog([paths[name] for name in sorted(development)])
    catalog_path = args.out_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    parts = []
    with gzip.open(args.out_dir / "spans.jsonl.gz", "wt", encoding="utf-8") as output:
        for name, path in sorted(paths.items()):
            part = evaluate_pdf(path, catalog, output)
            part["split"] = "development" if name in development else "holdout"
            parts.append(part)
    summary = {}
    for label in ("development", "holdout"):
        selected = [part for part in parts if part["split"] == label]
        summary[label] = {
            key: sum(part.get(key, 0) for part in selected)
            for key in ("source_mapping", "matched_outline", "unresolved")
        }
        summary[label]["pdfs"] = len(selected)
    audit = {
        "summary": summary,
        "parts": parts,
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "pymupdf_version": __import__("pymupdf").VersionBind,
        "limitations": [
            "Counts measure physical glyph groups, not names or elector records.",
            "Recovered strings remain in visual order; they are not training inputs.",
            "Only exact source outlines with unique mappings are reused.",
            "No empirical character-accuracy estimate is claimed.",
        ],
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
