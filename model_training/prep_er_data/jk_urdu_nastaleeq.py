"""Recover candidate Nastaleeq name fields from verified PDF outlines.

The card adapter verifies the printed voter-name label, expands complete text
operators to retain displaced dots, and rejects mixed lines and unknown glyphs.
The identity ledger is supplied by the structural parser and is never inferred
from decoded names.
"""

import hashlib
import io
from collections import defaultdict
from itertools import pairwise, product

import numpy as np
from fontTools.ttLib import TTFont

from model_training.prep_er_data.hindi_glyphs import glyph_groups
from model_training.prep_er_data.jk_urdu_unicode import (
    RELATIONSHIPS,
    native_text,
    outline_key,
)
from model_training.prep_er_data.urdu_text import MAX_CANDIDATES


def inside(box, origin):
    """Check a physical origin against a fixed source rectangle."""
    return box[0] <= origin[0] <= box[2] and box[1] <= origin[1] <= box[3]


class IndexedPage(list):
    """Keep original drawing order and cache exact inclusive rectangle queries."""

    def __init__(self, values):
        """Index page glyphs and prepare a rectangle-query cache."""
        super().__init__(dict(g, index=i) for i, g in enumerate(values))
        self.positions = np.asarray([g["origin"] for g in self])
        self.cards = {}

    def card(self, box):
        """Return glyphs whose physical origins lie inside a card box."""
        key = tuple(box)
        if key not in self.cards:
            if not self:
                self.cards[key] = []
            else:
                x, y = self.positions.T
                mask = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
                self.cards[key] = [self[i] for i in np.flatnonzero(mask)]
        return self.cards[key]


class PdfOutlines:
    """Resolve physical glyphs across the embedded subsets used on each page."""

    def __init__(self, document, reference):
        """Initialize embedded-font caches for an open PDF document."""
        self.document = document
        self.reference = reference
        self.fonts = {}
        self.resolutions = {}

    def page(self, page):
        """Read every physical glyph, retaining unresolved glyphs as explicit errors."""
        by_name = defaultdict(set)
        for xref, extension, _, name, *_ in page.get_fonts():
            if extension != "ttf":
                by_name[name.split("+")[-1]].add(None)
                continue
            if xref not in self.fonts:
                data = self.document.extract_font(xref)[3]
                self.fonts[xref] = (
                    hashlib.sha256(data).hexdigest(),
                    TTFont(io.BytesIO(data)),
                )
            by_name[name.split("+")[-1]].add(xref)
        output = []
        for span in page.get_texttrace():
            namespace = frozenset(by_name[span["font"]])
            for group in glyph_groups(span["chars"]):
                key = (namespace, group["gid"], group["text"] == " ")
                if key not in self.resolutions:
                    choices = {}
                    for xref in namespace:
                        if xref is None:
                            choices["unsupported_font"] = None
                            continue
                        digest, font = self.fonts[xref]
                        if "glyf" not in font or font["head"].unitsPerEm != 2048:
                            choices["unsupported_font"] = None
                            continue
                        gid = group["gid"]
                        if not 0 < gid < len(font.getGlyphOrder()):
                            continue
                        identity = outline_key(font, gid)
                        if identity:
                            choices[identity] = (font, gid, digest)
                        elif group["text"] == " ":
                            choices["space"] = None
                    if len(choices) != 1:
                        value = {"issue": "ambiguous_or_missing_outline", "key": None}
                    else:
                        identity, source = next(iter(choices.items()))
                        value = {"issue": None, "key": identity, "outline": None}
                        if identity == "unsupported_font":
                            value["issue"] = identity
                        elif source is not None:
                            font, gid, digest = source
                            try:
                                value["outline"] = self.reference.source_outline(
                                    font, gid
                                )
                                value["font_sha256"] = digest
                            except ValueError as error:
                                value["issue"] = str(error)
                    self.resolutions[key] = value
                output.append(
                    {
                        **group,
                        **self.resolutions[key],
                        "size": span["size"],
                        "dir": tuple(span["dir"]),
                        "seqno": span["seqno"],
                    }
                )
        return IndexedPage(output)


def field_groups(box, page_glyphs, colon_key):
    """Select a whole own-name field and its label, rejecting uncertain boundaries."""
    width, height = box[2] - box[0], box[3] - box[1]
    if not (175 <= width <= 185 and 61 <= height <= 71):
        return None, "unsupported_card_geometry"
    glyphs = (
        page_glyphs.card(box)
        if isinstance(page_glyphs, IndexedPage)
        else [g for g in page_glyphs if inside(box, g["origin"])]
    )
    own = [
        g
        for g in glyphs
        if g["key"] == colon_key and 17 <= g["origin"][1] - box[1] <= 35
    ]
    following = [
        g
        for g in glyphs
        if g["key"] == colon_key and 35 < g["origin"][1] - box[1] <= 50
    ]
    if len(own) != 1 or len(following) != 1:
        return None, "missing_or_ambiguous_line_separator"
    separator = own[0]
    x, y = separator["origin"]
    end = (y + following[0]["origin"][1]) / 2
    # The seed identifies text operators, not the final set of name glyphs.
    seed = [
        g
        for g in glyphs
        if box[1] + 17 <= g["origin"][1] <= box[1] + 35 and g["origin"][0] < x - 0.01
    ]
    operators = {g["seqno"] for g in seed}
    selected = [
        g for g in glyphs if g["seqno"] in operators and g["origin"][0] < x - 0.01
    ]
    if not selected:
        return None, "empty_name"
    if any(g["origin"][1] >= end or g["origin"][1] < box[1] + 12 for g in selected):
        return None, "operator_crosses_field_boundary"
    omitted = [
        g
        for g in glyphs
        if g["origin"][0] < x - 0.01
        and box[1] + 17 <= g["origin"][1] < end
        and g["seqno"] not in operators
    ]
    if any(g["key"] != "space" for g in omitted):
        return None, "unassigned_name_line_glyph"
    labels = [
        g
        for g in glyphs
        if x + 0.01 < g["origin"][0] and box[1] + 17 <= g["origin"][1] <= box[1] + 35
    ]
    label_operators = {g["seqno"] for g in labels}
    labels = [
        g for g in glyphs if g["seqno"] in label_operators and g["origin"][0] > x + 0.01
    ]
    if any(g["origin"][1] >= end or g["origin"][1] < box[1] + 12 for g in labels):
        return None, "label_operator_crosses_field_boundary"
    return {
        "name": selected,
        "label": labels,
        "separator": separator,
        "bottom": end,
    }, None


def decode_line(groups, reference):
    """Decode complete words separated by explicitly blank source-space glyphs."""
    if not groups or any(g["issue"] for g in groups):
        return (), "unresolved_physical_glyph"
    sizes = {g["size"] for g in groups}
    if len(sizes) != 1 or next(iter(sizes)) <= 0:
        return (), "mixed_or_invalid_size"
    if any(g["dir"] != (1.0, 0.0) for g in groups):
        return (), "unsupported_direction"
    scale = next(iter(sizes)) / 2048
    words = [[]]
    for group in groups:
        if group["key"] == "space":
            if words[-1]:
                words.append([])
        else:
            x, y = group["origin"]
            words[-1].append((group["outline"], x / scale, -y / scale))
    words = [word for word in words if word]
    if not words:
        return (), "empty_name"
    if any(
        min(g[1] for g in left) >= min(g[1] for g in right)
        for left, right in pairwise(words)
    ):
        return (), "nonmonotone_word_positions"
    try:
        decoded = [reference.decode(word) for word in words]
    except ValueError:
        return (), "candidate_limit"
    count = 1
    for candidates in decoded:
        count *= len(candidates)
        if count > MAX_CANDIDATES:
            return (), "candidate_limit"
    if not count:
        return (), "unresolved_word"
    candidates = tuple(sorted(" ".join(words) for words in product(*reversed(decoded))))
    return candidates, None


def select_name_field(box, page_glyphs, colon_key, reference, image_boxes=()):
    """Select a complete name field after its boundaries and label pass."""
    fields, issue = field_groups(box, page_glyphs, colon_key)
    if issue:
        return {"candidate": None, "issue": issue}
    name_region = (
        box[0],
        box[1] + 12,
        fields["separator"]["origin"][0],
        fields["bottom"],
    )
    if any(
        b[0] < name_region[2]
        and b[2] > name_region[0]
        and b[1] < name_region[3]
        and b[3] > name_region[1]
        for b in image_boxes
    ):
        return {"candidate": None, "issue": "name_image_overlap"}
    labels, issue = decode_line(fields["label"], reference)
    if issue or not any(label.replace(" ", "") == "نامووٹر" for label in labels):
        return {"candidate": None, "issue": "unverified_voter_name_label"}
    return {"groups": fields["name"], "region": name_region, "issue": None}


def recover_name(box, page_glyphs, colon_key, reference, image_boxes=()):
    """Decode a selected name while retaining every field and label guard."""
    selected = select_name_field(box, page_glyphs, colon_key, reference, image_boxes)
    if selected["issue"]:
        return {"candidate": None, "issue": selected["issue"]}
    candidates, issue = decode_line(selected["groups"], reference)
    if issue:
        return {"candidate": None, "issue": issue}
    if len(candidates) != 1:
        return {
            "candidate": None,
            "issue": "ambiguous_unicode",
            "alternatives": candidates,
        }
    text = candidates[0]
    if native_text(text) != text:
        return {"candidate": None, "issue": "unsupported_name_text"}
    return {"candidate": text, "issue": None}


def recover_part(rows, source, reference, colon_key):
    """Enrich one immutable source partition without altering its identity ledger."""
    import time
    from collections import Counter

    import pymupdf

    started = time.monotonic()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if not rows or any(
        row["filename"] != source.name or row["source_sha256"] != digest for row in rows
    ):
        raise ValueError("Inventory identity does not match its source PDF")
    if len({row["event_key"] for row in rows}) != len(rows):
        raise ValueError("Inventory repeats a source event key")
    by_page = defaultdict(list)
    for index, row in enumerate(rows):
        by_page[row["page"]].append((index, row))
    output = [dict(row) for row in rows]
    issues = Counter()
    with pymupdf.open(source) as document:
        reader = PdfOutlines(document, reference)
        for number, page_rows in by_page.items():
            page = document[number - 1]
            glyphs = reader.page(page)
            images = [image["bbox"] for image in page.get_image_info()]
            for index, row in page_rows:
                if row["elector_name"] is not None:
                    issues["existing_name_preserved"] += 1
                    continue
                result = recover_name(row["bbox"], glyphs, colon_key, reference, images)
                issues[result["issue"] or "verified_geometry"] += 1
                output[index]["elector_name"] = result["candidate"]
                output[index]["name_issue"] = result["issue"]
                if result["candidate"] is not None:
                    output[index]["name_candidate"] = result["candidate"]
    return output, {
        "filename": source.name,
        "source_sha256": digest,
        "inventory": len(rows),
        "active": sum(row["active"] for row in output),
        "accepted_names": sum(row["elector_name"] is not None for row in output),
        "accepted_active_names": sum(
            row["active"] and row["elector_name"] is not None for row in output
        ),
        "name_issue_counts": dict(issues),
        "elapsed_seconds": time.monotonic() - started,
    }


_CONTROLS = {"خانہ", "خانہنمبر", "عمر", "جنس"}


def _label_before(glyphs, card, separator, allowed, reference):
    """Verify the complete fixed label immediately preceding a separator."""
    x, y = separator["origin"]
    card_indices = {g["index"] for g in card}
    end = separator["index"]
    start = end
    while start > 0:
        previous = glyphs[start - 1]
        blank = previous["key"] == "space"
        if (
            previous["index"] not in card_indices
            or previous["origin"][0] <= x + (0.01 if not blank else -0.01)
            or not y - 14 <= previous["origin"][1] <= y + 6
        ):
            break
        start -= 1
        if end - start > 64:
            return None
    if start == end:
        return None
    selected = glyphs[start:end]
    values, issue = decode_line(selected, reference)
    found = {value.replace(" ", "") for value in values} & allowed
    if issue or len(found) != 1:
        return None
    return {"start": start, "end": end, "text": next(iter(found))}


def select_relative_field(
    box, page_glyphs, colon_key, label_reference, identity, image_boxes=()
):
    """Select a relative field and preserve its verified printed relationship."""

    def failure(issue, relationship=None):
        return {"candidate": None, "issue": issue, "relationship": relationship}

    width, height = box[2] - box[0], box[3] - box[1]
    if not (175 <= width <= 185 and 61 <= height <= 71):
        return failure("unsupported_card_geometry")
    glyphs = (
        page_glyphs
        if isinstance(page_glyphs, IndexedPage)
        else IndexedPage(page_glyphs)
    )
    card = glyphs.card(box)
    cs = [g for g in card if g["key"] == colon_key]
    own = [g for g in cs if 17 <= g["origin"][1] - box[1] <= 35]
    rel = [g for g in cs if 35 < g["origin"][1] - box[1] <= 50]
    lower = [g for g in cs if 50 < g["origin"][1] - box[1] <= 66]
    if len(own) != 1 or len(rel) != 1:
        return failure("missing_or_ambiguous_line_separator")
    own, rel = own[0], rel[0]
    rel_label = _label_before(glyphs, card, rel, RELATIONSHIPS, label_reference)
    if rel_label is None:
        return failure("unverified_relationship_label")
    if not lower:
        return failure("missing_following_control")
    following = min(lower, key=lambda g: g["index"])
    label = _label_before(glyphs, card, following, _CONTROLS, label_reference)
    if label is None:
        return failure("unverified_following_control_label")
    separator = rel
    if separator["index"] >= label["start"]:
        return failure("unsupported_field_drawing_order")
    selected = glyphs[separator["index"] + 1 : label["start"]]
    if identity and all(value is not None for value in identity):
        epic, number = identity
        expected = {str(epic) + str(number), str(number) + str(epic)}
        header = [g for g in selected if 0 <= g["origin"][1] - box[1] < 17]
        header_indices = {g["index"] for g in header}
        if header and (
            all(inside(box, g["origin"]) and g["text"].isascii() for g in header)
            and "".join(g["text"] for g in header).replace(" ", "") in expected
        ):
            selected = [g for g in selected if g["index"] not in header_indices]
    if not selected:
        return failure("empty_name")
    x, y = separator["origin"]
    end = following["origin"][1] - 0.5
    if any(
        not inside(box, g["origin"])
        or g["origin"][0] >= x - 0.01
        or not y - 14 <= g["origin"][1] < end
        for g in selected
    ):
        return failure("name_stream_crosses_field_boundary")
    region = (box[0], y - 14, x, end)
    if any(
        b[0] < region[2] and b[2] > region[0] and b[1] < region[3] and b[3] > region[1]
        for b in image_boxes
    ):
        return failure("name_image_overlap")
    return {
        "groups": selected,
        "region": region,
        "issue": None,
        "relationship": rel_label["text"],
    }


def recover_relative(
    box, page_glyphs, colon_key, reference, label_reference, identity, image_boxes=()
):
    """Decode a guarded relative name without discarding its verified label."""
    selected = select_relative_field(
        box, page_glyphs, colon_key, label_reference, identity, image_boxes
    )
    if selected["issue"]:
        return {"candidate": None, "issue": selected["issue"], "relationship": None}
    values, issue = decode_line(selected["groups"], reference)
    if issue is None and len(values) != 1:
        issue = "ambiguous_unicode"
    if issue is None and native_text(values[0]) != values[0]:
        issue = "unsupported_name_text"
    return {
        "candidate": values[0] if issue is None else None,
        "issue": issue,
        "relationship": selected["relationship"],
    }


def recover_relative_part(rows, source, reference, label_reference, colon_key):
    """Enrich relative fields while preserving prior names and the identity ledger."""
    import time
    from collections import Counter

    import pymupdf

    started = time.monotonic()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if not rows or any(
        row["filename"] != source.name or row["source_sha256"] != digest for row in rows
    ):
        raise ValueError("Inventory identity does not match its source PDF")
    if len({row["event_key"] for row in rows}) != len(rows):
        raise ValueError("Inventory repeats a source event key")
    by_page = defaultdict(list)
    for index, row in enumerate(rows):
        by_page[row["page"]].append((index, row))
    output = [dict(row) for row in rows]
    issues = Counter()
    with pymupdf.open(source) as document:
        reader = PdfOutlines(document, reference)
        for number, page_rows in by_page.items():
            page = document[number - 1]
            glyphs = reader.page(page)
            images = [image["bbox"] for image in page.get_image_info()]
            for index, row in page_rows:
                if row["relative_name"] is not None:
                    issues["existing_relative_preserved"] += 1
                    continue
                result = recover_relative(
                    row["bbox"],
                    glyphs,
                    colon_key,
                    reference,
                    label_reference,
                    (row["id"], row["number"]),
                    images,
                )
                issues[result["issue"] or "verified_geometry"] += 1
                output[index]["relative_name"] = result["candidate"]
                output[index]["relative_issue"] = result["issue"]
                if result["candidate"] is not None:
                    output[index]["relative_candidate"] = result["candidate"]
                if result["relationship"] is not None and (
                    result["candidate"] is not None or row["relative_type"] is None
                ):
                    output[index]["relative_type"] = result["relationship"]
    return output, {
        "filename": source.name,
        "source_sha256": digest,
        "inventory": len(rows),
        "active": sum(row["active"] for row in output),
        "accepted_relatives": sum(row["relative_name"] is not None for row in output),
        "accepted_active_relatives": sum(
            row["active"] and row["relative_name"] is not None for row in output
        ),
        "relative_issue_counts": dict(issues),
        "elapsed_seconds": time.monotonic() - started,
    }
