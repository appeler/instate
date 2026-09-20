"""Normalize exact Urdu word outlines while retaining every source glyph."""

import hashlib
import json
from collections import defaultdict
from itertools import pairwise

QUANTUM = 8
REVISION = "jk-word-outlines-q8-v1"
OCR_MINIMUM_RECORDS_AND_CALLS = 6
OCR_MINIMUM_SHARE = 0.75
OCR_MINIMUM_MARGIN = 1


def select_supported_reading(
    evidence,
    minimum=OCR_MINIMUM_RECORDS_AND_CALLS,
    minimum_share=OCR_MINIMUM_SHARE,
    minimum_margin=OCR_MINIMUM_MARGIN,
):
    """Select one reading from independently sourced OCR evidence.

    Support is the smaller of the distinct-record and distinct-call counts.
    The winning reading must also clear the requested share in both dimensions
    and exceed the runner-up by the requested support margin.
    """
    if minimum < 1 or not 0 < minimum_share <= 1 or minimum_margin < 1:
        raise ValueError("Invalid OCR support policy")
    records = defaultdict(set)
    calls = defaultdict(set)
    all_records = set()
    all_calls = set()
    for item in evidence:
        reading = item["reading"]
        record = item["entry_event_key"]
        call = item["call"]
        records[reading].add(record)
        calls[reading].add(call)
        all_records.add(record)
        all_calls.add(call)
    if not records:
        return None
    ranked = sorted(
        (
            min(len(source_records), len(calls[reading])),
            min(
                len(source_records) / len(all_records),
                len(calls[reading]) / len(all_calls),
            ),
            reading,
        )
        for reading, source_records in records.items()
    )
    support, share, reading = ranked[-1]
    runner_up = ranked[-2][0] if len(ranked) > 1 else 0
    margin = support - runner_up
    if support < minimum or share < minimum_share or margin < minimum_margin:
        return None
    return {
        "reading": reading,
        "distinct_records": len(records[reading]),
        "distinct_calls": len(calls[reading]),
        "support": support,
        "share": share,
        "margin": margin,
    }


def word_drawings(groups):
    """Return exact and normalized keys inside already verified name boundaries."""
    if not groups:
        return None, "empty_name"
    if any(g["key"] in (None, "unsupported_font") for g in groups):
        return None, "unresolved_physical_glyph"
    sizes = {g["size"] for g in groups}
    if len(sizes) != 1 or min(sizes) <= 0:
        return None, "mixed_or_invalid_size"
    if any(g["dir"] != (1.0, 0.0) for g in groups):
        return None, "unsupported_direction"
    chunks = [[]]
    for group in groups:
        if group["key"] == "space":
            if chunks[-1]:
                chunks.append([])
        else:
            chunks[-1].append(group)
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return None, "empty_name"
    if any(
        min(g["origin"][0] for g in left) >= min(g["origin"][0] for g in right)
        for left, right in pairwise(chunks)
    ):
        return None, "nonmonotone_word_positions"
    result = []
    for chunk in reversed(chunks):
        x = min(g["origin"][0] for g in chunk)
        y = min(g["origin"][1] for g in chunk)
        drawing = sorted(
            (
                g["key"],
                round((g["origin"][0] - x) * 10000),
                round((g["origin"][1] - y) * 10000),
                round(g["size"] * 10000),
            )
            for g in chunk
        )
        if any(size <= 0 for _, _, _, size in drawing):
            return None, "mixed_or_invalid_size"
        geometry = json.dumps(drawing, separators=(",", ":")).encode()
        normalized = sorted(
            (
                glyph,
                round(dx * 2048 / size / QUANTUM),
                round(dy * 2048 / size / QUANTUM),
            )
            for glyph, dx, dy, size in drawing
        )
        result.append(
            {
                "original": hashlib.sha256(geometry).digest(),
                "normalized": hashlib.sha256(
                    json.dumps(normalized, separators=(",", ":")).encode()
                ).digest(),
                "geometry": geometry,
            }
        )
    return result, None
