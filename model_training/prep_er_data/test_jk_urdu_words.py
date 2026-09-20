"""Check scale reuse without losing glyph identity, dots or word boundaries."""

import pytest

from model_training.prep_er_data.jk_urdu_words import (
    select_supported_reading,
    word_drawings,
)


def glyph(key, x, y, size=10):
    return {"key": key, "origin": (x, y), "size": size, "dir": (1.0, 0.0)}


def test_scaled_printing_reuses_words_but_retains_original_drawings():
    small, issue = word_drawings([glyph("body", 10, 10), glyph("dot", 15, 12)])
    large, other_issue = word_drawings(
        [glyph("body", 40, 30, 20), glyph("dot", 50, 34, 20)]
    )
    assert issue is other_issue is None
    assert small[0]["original"] != large[0]["original"]
    assert small[0]["normalized"] == large[0]["normalized"]


@pytest.mark.parametrize("change", ["missing_dot", "different_body", "moved_dot"])
def test_letter_and_dot_changes_do_not_collapse(change):
    source = [glyph("body", 10, 10), glyph("dot", 15, 12)]
    original, _ = word_drawings(source)
    if change == "missing_dot":
        source.pop()
    elif change == "different_body":
        source[0]["key"] = "other_body"
    else:
        source[1]["origin"] = (15, 12.5)
    changed, _ = word_drawings(source)
    assert original[0]["normalized"] != changed[0]["normalized"]


def test_tiny_placement_difference_keeps_original_evidence():
    a, _ = word_drawings([glyph("body", 10, 10), glyph("dot", 15, 12)])
    b, _ = word_drawings([glyph("body", 10, 10), glyph("dot", 15.001, 12)])
    assert a[0]["normalized"] == b[0]["normalized"]
    assert a[0]["geometry"] != b[0]["geometry"]


def test_only_explicit_spaces_define_words():
    groups = [glyph("left", 10, 10), glyph("space", 15, 10), glyph("right", 20, 10)]
    split, issue = word_drawings(groups)
    assert issue is None
    assert len(split) == 2
    assert split[0] == word_drawings([groups[-1]])[0][0]
    groups[1]["key"] = "visible_glyph"
    joined, issue = word_drawings(groups)
    assert issue is None
    assert len(joined) == 1


@pytest.mark.parametrize("key", [None, "unsupported_font"])
def test_unresolved_glyph_is_not_dropped(key):
    assert word_drawings([glyph("body", 10, 10), glyph(key, 15, 12)])[1] == (
        "unresolved_physical_glyph"
    )


def test_mixed_sizes_and_reversed_word_order_are_withheld():
    assert word_drawings([glyph("a", 10, 10), glyph("b", 20, 10, 12)])[1] == (
        "mixed_or_invalid_size"
    )
    assert (
        word_drawings([glyph("a", 20, 10), glyph("space", 15, 10), glyph("b", 10, 10)])[
            1
        ]
        == "nonmonotone_word_positions"
    )


def reading(value, record, call):
    return {"reading": value, "entry_event_key": record, "call": call}


def test_ocr_vote_requires_six_records_calls_and_three_quarters_agreement():
    evidence = [reading("الف", f"r{i}", i) for i in range(6)]
    assert select_supported_reading(evidence) == {
        "reading": "الف",
        "distinct_records": 6,
        "distinct_calls": 6,
        "support": 6,
        "share": 1.0,
        "margin": 6,
    }
    assert select_supported_reading(evidence[:5]) is None
    evidence.extend(reading("ب", f"x{i}", i + 6) for i in range(2))
    assert select_supported_reading(evidence)["reading"] == "الف"
    evidence.append(reading("ب", "x2", 8))
    assert select_supported_reading(evidence) is None


def test_ocr_vote_counts_records_and_calls_independently():
    same_call = [reading("الف", f"r{i}", 1) for i in range(6)]
    assert select_supported_reading(same_call) is None
    same_record = [reading("الف", "r", i) for i in range(6)]
    assert select_supported_reading(same_record) is None


@pytest.mark.parametrize(
    ("minimum", "share", "margin"), [(0, 0.75, 1), (6, 0, 1), (6, 1.1, 1), (6, 0.75, 0)]
)
def test_ocr_vote_rejects_invalid_policy(minimum, share, margin):
    with pytest.raises(ValueError, match="Invalid OCR support policy"):
        select_supported_reading([], minimum, share, margin)
