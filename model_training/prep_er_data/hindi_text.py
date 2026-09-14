"""Decode Mangal outlines and verify Hindi ordering by reshaping the result."""

import hashlib
import itertools
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from model_training.prep_er_data.hindi_glyphs import glyph_fingerprint, usable_unicode

REPH = "\ue000"
CONSONANT = "[\u0915-\u0939\u0958-\u095f]"
CLUSTER = rf"{CONSONANT}\u093c?(?:\u094d{CONSONANT}\u093c?)*"
MARKS = "[\u0900-\u0903\u093e-\u094c\u0962\u0963]*"


def logical_order(text):
    """Reverse pre-base i and explicitly tagged reph within consonant clusters."""
    text = re.sub(rf"ि({REPH}?)({CLUSTER})", r"\1\2ि", text)
    text = re.sub(rf"({CLUSTER}{MARKS}){REPH}", r"र्\1", text)
    return unicodedata.normalize("NFC", text)


def reverse_substitutions(seed, rules, reph_rules):
    """Collect all reachable meanings; ambiguity propagates to dependent glyphs."""
    candidates = defaultdict(set, {key: set(value) for key, value in seed.items()})
    for _ in range(len(rules) + 1):
        before = sum(map(len, candidates.values()))
        for sources, target, index in rules:
            if not all(candidates[source] for source in sources):
                continue
            for values in itertools.product(
                *(candidates[source] for source in sources)
            ):
                text = "".join(values)
                if index in reph_rules and text == "र्":
                    text = REPH
                if len(text) > 32 or len(candidates[target]) > 64:
                    raise ValueError(
                        "Unbounded or excessively ambiguous GSUB inversion"
                    )
                candidates[target].add(text)
        if before == sum(map(len, candidates.values())):
            return dict(candidates)
    raise ValueError("GSUB inversion did not converge")


def font_mapping(font):
    """Invert only reachable modern Devanagari substitutions from the font."""
    seed = defaultdict(set)
    for codepoint, glyph in (font.getBestCmap() or {}).items():
        if usable_unicode(chr(codepoint)) or codepoint == 0x25CC:
            seed[glyph].add(unicodedata.normalize("NFD", chr(codepoint)))
    table = font["GSUB"].table
    scripts = {
        record.ScriptTag: record.Script for record in table.ScriptList.ScriptRecord
    }
    if "dev2" not in scripts or scripts["dev2"].DefaultLangSys is None:
        raise ValueError("Reference font must include modern Devanagari shaping")
    language_systems = [scripts["dev2"].DefaultLangSys]
    language_systems.extend(record.LangSys for record in scripts["dev2"].LangSysRecord)
    feature_indices = {
        index for language in language_systems for index in language.FeatureIndex
    }
    features = [
        table.FeatureList.FeatureRecord[index] for index in sorted(feature_indices)
    ]
    active, reph_rules = set(), set()

    def visit(index, selected):
        if index in selected:
            return
        selected.add(index)
        lookup = table.LookupList.Lookup[index]
        for subtable in lookup.SubTable:
            if lookup.LookupType == 6:
                if subtable.Format != 3:
                    raise ValueError("Unsupported contextual GSUB format")
                for record in subtable.SubstLookupRecord:
                    visit(record.LookupListIndex, selected)
            elif lookup.LookupType not in (1, 4):
                raise ValueError(f"Unsupported GSUB lookup type: {lookup.LookupType}")

    for feature in features:
        for index in feature.Feature.LookupListIndex:
            visit(index, active)
            if feature.FeatureTag == "rphf":
                visit(index, reph_rules)
    rules = []
    for index in sorted(active):
        lookup = table.LookupList.Lookup[index]
        for subtable in lookup.SubTable:
            if lookup.LookupType == 1:
                rules.extend(
                    ((source,), target, index)
                    for source, target in subtable.mapping.items()
                )
            elif lookup.LookupType == 4:
                for first, ligatures in subtable.ligatures.items():
                    rules.extend(
                        ((first, *ligature.Component), ligature.LigGlyph, index)
                        for ligature in ligatures
                    )
    candidates = reverse_substitutions(seed, rules, reph_rules)
    return {
        font.getGlyphID(glyph): values for glyph, values in candidates.items() if values
    }


class ReferenceFonts:
    """Match exact outlines to font-defined text; verify complete word rendering."""

    def __init__(self, paths):
        import uharfbuzz as hb
        from fontTools.ttLib import TTFont

        self.mapping = {}
        self.fonts = []
        self.sources = []
        candidates = defaultdict(set)
        for path in paths:
            path = Path(path)
            data = path.read_bytes()
            with TTFont(path) as font:
                meanings = font_mapping(font)
                fingerprints = {
                    gid: glyph_fingerprint(font, gid)
                    for gid in range(len(font.getGlyphOrder()))
                }
                for gid, texts in meanings.items():
                    fingerprint = fingerprints[gid]
                    if fingerprint:
                        candidates[fingerprint].update(texts)
                self.sources.append(
                    {
                        "filename": path.name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "version": font["name"].getDebugName(5),
                        "glyphs": len(font.getGlyphOrder()),
                    }
                )
            self.fonts.append((hb.Font(hb.Face(data)), fingerprints))
        self.mapping = {
            key: next(iter(values))
            for key, values in candidates.items()
            if len(values) == 1
        }
        self.conflicts = {
            key: sorted(values) for key, values in candidates.items() if len(values) > 1
        }
        self._shape_cache = {}

    def decode(self, group, fingerprint):
        if fingerprint in self.mapping:
            return self.mapping[fingerprint], "reference_outline"
        if fingerprint in self.conflicts:
            return "\ufffd", "ambiguous_outline"
        if usable_unicode(group["text"]):
            return group["text"], "source_mapping"
        return "\ufffd", "unresolved"

    def verifies(self, text, fingerprints):
        """Require identical nonblank glyph outlines in identical drawing order."""
        if not text or "\ufffd" in text or REPH in text or None in fingerprints:
            return False
        import uharfbuzz as hb

        if text not in self._shape_cache:
            rendered = []
            for font, mapping in self.fonts:
                for language in ("hi", "sa"):
                    buffer = hb.Buffer()
                    buffer.add_str(text)
                    buffer.guess_segment_properties()
                    buffer.language = language
                    hb.shape(font, buffer)
                    rendered.append(
                        tuple(mapping[info.codepoint] for info in buffer.glyph_infos)
                    )
            self._shape_cache[text] = rendered
        return tuple(fingerprints) in self._shape_cache[text]


def word_groups(glyphs, line_tolerance=2.5, word_gap=2.0):
    """Separate positioned glyphs into words without trusting PDF span boundaries."""

    def horizontal_key(glyph):
        # Zero-advance marks can extend just beyond a following space origin.
        mark = all(
            unicodedata.category(char).startswith("M") or char == REPH
            for char in glyph["decoded"]
        )
        offset = 0.25 if mark else 0
        return glyph["origin"][0] - offset, glyph["ordinal"]

    lines = []
    for glyph in sorted(glyphs, key=lambda value: value["origin"][1]):
        y = glyph["origin"][1]
        if not lines or y - lines[-1][0] > line_tolerance:
            lines.append((y, []))
        lines[-1][1].append(glyph)
    for _, line in lines:
        word, right = [], None
        for glyph in sorted(line, key=horizontal_key):
            x = glyph["origin"][0]
            separator = glyph["decoded"].isspace()
            if separator or (right is not None and x - right > word_gap):
                if word:
                    yield word
                word, right = [], None
            if not separator:
                word.append(glyph)
                right = max(right or x, glyph["bbox"][2])
        if word:
            yield word


def extract_page(page, outlines, references):
    """Preserve source glyphs, recover mappings and gate reordered word candidates."""
    from model_training.prep_er_data.hindi_glyphs import glyph_groups

    glyphs = []
    for span in page.get_texttrace():
        if "Mangal" not in span["font"]:
            continue
        mapping = outlines.get(span["font"], {})
        for group in glyph_groups(span["chars"]):
            fingerprint = mapping.get(group["gid"])
            decoded, status = references.decode(group, fingerprint)
            glyphs.append(
                {
                    **group,
                    "font": span["font"],
                    "fingerprint": fingerprint,
                    "decoded": decoded,
                    "status": status,
                    "ordinal": len(glyphs),
                }
            )
    words = []
    for word in word_groups(glyphs):
        candidate = logical_order("".join(glyph["decoded"] for glyph in word))
        fingerprints = [glyph["fingerprint"] for glyph in word]
        verified = references.verifies(candidate, fingerprints)
        words.append(
            {
                "raw_text": "".join(glyph["text"] for glyph in word),
                "candidate": candidate,
                "logical_text": candidate if verified else None,
                "render_verified": verified,
                "bbox": [
                    min(glyph["bbox"][0] for glyph in word),
                    min(glyph["bbox"][1] for glyph in word),
                    max(glyph["bbox"][2] for glyph in word),
                    max(glyph["bbox"][3] for glyph in word),
                ],
                "glyphs": word,
            }
        )
    return glyphs, words


def main():
    import argparse
    import gzip
    import json
    import time
    from collections import Counter
    from importlib.metadata import version

    import pymupdf

    from model_training.prep_er_data.hindi_glyphs import FontOutlines

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, action="append", required=True)
    parser.add_argument("--reference-font", type=Path, action="append", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    parts = [(label, name) for label, names in split.items() for name in names]
    if len({name for _, name in parts}) != len(parts):
        parser.error("PDF filenames must be unique across split groups")
    paths = {}
    hashes = set()
    for _, name in parts:
        found = [root / name for root in args.pdf_dir if (root / name).is_file()]
        if len(found) != 1:
            parser.error(f"Expected one source PDF for {name}, found {len(found)}")
        digest = hashlib.sha256(found[0].read_bytes()).hexdigest()
        if digest in hashes:
            parser.error(f"Duplicate PDF content: {name}")
        hashes.add(digest)
        paths[name] = found[0]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    references = ReferenceFonts(args.reference_font)
    results = []
    with gzip.open(args.out_dir / "words.jsonl.gz", "wt", encoding="utf-8") as output:
        for label, name in parts:
            counts = Counter()
            with pymupdf.open(paths[name]) as document:
                fonts = FontOutlines(document)
                for page in document:
                    glyphs, words = extract_page(page, fonts.page(page), references)
                    for glyph in glyphs:
                        counts["glyph_groups"] += 1
                        counts[glyph["status"]] += 1
                        damaged = not usable_unicode(glyph["text"])
                        counts["source_damaged"] += damaged
                        counts["recovered_damaged"] += (
                            damaged and glyph["status"] == "reference_outline"
                        )
                    for word in words:
                        counts["words"] += 1
                        counts["render_verified_words"] += word["render_verified"]
                        output.write(
                            json.dumps(
                                {"filename": name, "page": page.number + 1, **word},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            result = {
                "filename": name,
                "split": label,
                "source_sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
                **counts,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    summary = {}
    for label in split:
        counts = Counter()
        selected = [part for part in results if part["split"] == label]
        for part in selected:
            counts.update(
                {key: value for key, value in part.items() if isinstance(value, int)}
            )
        summary[label] = {"pdfs": len(selected), **counts}
    audit = {
        "summary": summary,
        "parts": results,
        "reference_fonts": references.sources,
        "outline_mappings": len(references.mapping),
        "conflicting_outlines": references.conflicts,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outline_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("hindi_glyphs.py").read_bytes()
        ).hexdigest(),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "words_sha256": hashlib.sha256(
            (args.out_dir / "words.jsonl.gz").read_bytes()
        ).hexdigest(),
        "versions": {
            name: version(name) for name in ("PyMuPDF", "fonttools", "uharfbuzz")
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Word rendering agreement is not an empirical name-accuracy estimate.",
            "Unverified candidates remain separate from logical_text.",
            "Counts cover all Mangal text, not only names or elector records.",
            "Source spelling and malformed printed text are not corrected.",
            "The archive prefix is a convenience sample, not representative coverage.",
        ],
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
