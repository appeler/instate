"""Recover Urdu candidates from reference outlines and verify their shaping.

Nastaleeq substitutions can split one letter into a body and separate dots.
These fragments must be reunited before accepting a Unicode candidate. Glyph
numbers from a PDF are never treated as reference-font character mappings.
"""

import hashlib
import unicodedata
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from model_training.prep_er_data.hindi_glyphs import glyph_fingerprint

MAX_CANDIDATES = 4096
MAX_FRAGMENTS = 128


def compatible(fragments):
    """Check internal fragment boundaries, allowing incomplete outer edges."""
    for left, right in pairwise(fragments):
        char, index, length = left
        if index + 1 < length:
            if right != (char, index + 1, length):
                return False
        elif right[1] != 0:
            return False
    return True


def combine(left, right):
    """Compose compatible alternatives without silently truncating ambiguity."""
    result = set()
    for first in left:
        for second in right:
            value = first + second
            if len(value) > MAX_FRAGMENTS:
                raise ValueError("Urdu fragment sequence exceeds the recovery limit")
            if compatible(value):
                result.add(value)
            if len(result) > MAX_CANDIDATES:
                raise ValueError("Urdu candidates exceed the recovery limit")
    return result


def complete_text(fragments):
    """Return text only when every letter has all of its ordered fragments."""
    if not fragments or not compatible(fragments):
        return None
    if fragments[0][1] != 0 or fragments[-1][1] + 1 != fragments[-1][2]:
        return None
    return "".join(char for char, index, _ in fragments if index == 0)


def reverse_substitutions(seed, singles, multiples, ligatures):
    """Propagate Unicode fragments through supported font substitutions."""
    candidates = defaultdict(set)
    for glyph, texts in seed.items():
        candidates[glyph].update(tuple((char, 0, 1) for char in text) for text in texts)
    for _ in range(len(singles) + len(multiples) + len(ligatures) + 2):
        before = sum(map(len, candidates.values()))
        for source, target in singles:
            candidates[target].update(candidates[source])
        for source, targets in multiples:
            for value in tuple(candidates[source]):
                if len(targets) == 1:
                    candidates[targets[0]].add(value)
                elif len(value) == 1 and value[0][2] == 1:
                    for index, target in enumerate(targets):
                        candidates[target].add(((value[0][0], index, len(targets)),))
                else:
                    raise ValueError("Unsupported nested Urdu multiple substitution")
        for sources, target in ligatures:
            values = {()}
            for source in sources:
                values = combine(values, candidates[source])
            candidates[target].update(values)
        if any(len(values) > MAX_CANDIDATES for values in candidates.values()):
            raise ValueError("Urdu glyph meanings exceed the recovery limit")
        if before == sum(map(len, candidates.values())):
            return dict(candidates)
    raise ValueError("Urdu substitutions did not converge")


def font_mapping(font):
    """Derive candidates from Arabic Unicode mappings and reachable GSUB rules."""
    seed = defaultdict(set)
    for codepoint, glyph in (font.getBestCmap() or {}).items():
        if (
            codepoint == 32
            or 0x0600 <= codepoint <= 0x06FF
            or 0xFB50 <= codepoint <= 0xFDFF
            or 0xFE70 <= codepoint <= 0xFEFF
        ):
            seed[glyph].add(unicodedata.normalize("NFKC", chr(codepoint)))
    table = font["GSUB"].table
    scripts = {
        record.ScriptTag: record.Script for record in table.ScriptList.ScriptRecord
    }
    if "arab" not in scripts or scripts["arab"].DefaultLangSys is None:
        raise ValueError("Reference font must support Arabic shaping")
    systems = [scripts["arab"].DefaultLangSys]
    systems.extend(record.LangSys for record in scripts["arab"].LangSysRecord)
    features = {index for system in systems for index in system.FeatureIndex}
    features.update(
        system.ReqFeatureIndex for system in systems if system.ReqFeatureIndex != 65535
    )
    active = set()

    def visit(index):
        if index in active:
            return
        active.add(index)
        lookup = table.LookupList.Lookup[index]
        for subtable in lookup.SubTable:
            if lookup.LookupType == 6:
                if subtable.Format != 3:
                    raise ValueError("Unsupported contextual Urdu GSUB format")
                for record in subtable.SubstLookupRecord:
                    visit(record.LookupListIndex)
            elif lookup.LookupType not in (1, 2, 4):
                raise ValueError("Unsupported Urdu GSUB lookup type")

    for index in sorted(features):
        for lookup_index in table.FeatureList.FeatureRecord[
            index
        ].Feature.LookupListIndex:
            visit(lookup_index)
    singles, multiples, ligatures = [], [], []
    for index in sorted(active):
        lookup = table.LookupList.Lookup[index]
        for subtable in lookup.SubTable:
            if lookup.LookupType == 1:
                singles.extend(subtable.mapping.items())
            elif lookup.LookupType == 2:
                multiples.extend(subtable.mapping.items())
            elif lookup.LookupType == 4:
                for first, entries in subtable.ligatures.items():
                    ligatures.extend(
                        ((first, *entry.Component), entry.LigGlyph) for entry in entries
                    )
    candidates = reverse_substitutions(seed, singles, multiples, ligatures)
    return {
        font.getGlyphID(glyph): values for glyph, values in candidates.items() if values
    }


class ReferenceFont:
    """Invert exact reference outlines and retain only reshaping-consistent text."""

    def __init__(self, path):
        """Load a reference font with Unicode mappings and Arabic shaping rules."""
        import uharfbuzz as hb
        from fontTools.ttLib import TTFont

        path = Path(path)
        data = path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        with TTFont(path) as font:
            self.meanings = font_mapping(font)
            self.fingerprints = {
                gid: glyph_fingerprint(font, gid)
                for gid in range(len(font.getGlyphOrder()))
            }
        self.outlines = defaultdict(set)
        for gid, meanings in self.meanings.items():
            fingerprint = self.fingerprints[gid]
            if fingerprint:
                self.outlines[fingerprint].update(meanings)
        self.font = hb.Font(hb.Face(data))

    def shape(self, text):
        """Return outlines in drawing order, with None for unverified blank glyphs."""
        import uharfbuzz as hb

        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        buffer.language = "ur"
        hb.shape(self.font, buffer)
        return tuple(self.fingerprints[info.codepoint] for info in buffer.glyph_infos)

    def decode(self, fingerprints):
        """Return every verified candidate; callers must withhold ambiguous text."""
        fingerprints = tuple(fingerprints)
        if not fingerprints or any(value is None for value in fingerprints):
            return ()
        candidates = {()}
        for fingerprint in reversed(fingerprints):
            candidates = combine(candidates, self.outlines.get(fingerprint, ()))
            if not candidates:
                return ()
        texts = {text for value in candidates if (text := complete_text(value))}
        return tuple(sorted(text for text in texts if self.shape(text) == fingerprints))
