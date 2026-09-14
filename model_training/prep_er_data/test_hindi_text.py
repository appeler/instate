"""Regressions for font-defined Hindi recovery and positioned word assembly."""

import pytest

from model_training.prep_er_data.hindi_text import (
    REPH,
    ReferenceFonts,
    logical_order,
    reverse_substitutions,
    word_groups,
)


@pytest.mark.parametrize(
    ("visual", "logical"),
    [
        (f"शमा{REPH}", "शर्मा"),
        (f"िनवा{REPH}चक", "निर्वाचक"),
        ("िबहारी", "बिहारी"),
        ("ित्रवेदी", "त्रिवेदी"),
        (f"कीित{REPH}", "कीर्ति"),
        ("कृष्ण", "कृष्ण"),
        ("पुरुष", "पुरुष"),
        ("क\ufffdमा", "क\ufffdमा"),
        (f"{REPH} शर्मा", f"{REPH} शर्मा"),
    ],
)
def test_logical_order_preserves_letters_and_unknown_boundaries(visual, logical):
    assert logical_order(visual) == logical


def test_only_reph_feature_turns_ra_halant_into_reordering_marker():
    seed = {"ra": {"र"}, "halant": {"्"}}
    rules = [(("ra", "halant"), "reph", 3), (("ra", "halant"), "half_ra", 7)]
    result = reverse_substitutions(seed, rules, {3})
    assert result["reph"] == {REPH}
    assert result["half_ra"] == {"र्"}


def test_conflicting_meanings_propagate_through_ligatures():
    seed = {"a": {"क"}, "b": {"ख"}, "halant": {"्"}}
    rules = [
        (("a",), "shared", 1),
        (("b",), "shared", 1),
        (("shared", "halant"), "half", 4),
    ]
    result = reverse_substitutions(seed, rules, set())
    assert result["shared"] == {"क", "ख"}
    assert result["half"] == {"क्", "ख्"}


def test_recursive_gsub_expansion_fails_instead_of_truncating_candidates():
    with pytest.raises(ValueError):
        reverse_substitutions({"a": {"क"}}, [(("a", "a"), "a", 4)], set())


def positioned(text, x, y, width, ordinal):
    return {
        "decoded": text,
        "origin": (x, y),
        "bbox": (x, y - 8, x + width, y + 2),
        "ordinal": ordinal,
    }


def test_span_boundaries_and_vertical_mark_offsets_do_not_split_a_word():
    glyphs = [
        positioned("क", 10, 84, 6, 0),
        positioned("ु", 16, 85.8, 0, 1),
        positioned("मा", 16, 84, 9, 2),
        positioned("र", 25, 84, 5, 3),
        positioned(" ", 30, 84, 4, 4),
        positioned("देवी", 34, 84, 15, 5),
        positioned("पिता", 10, 103, 16, 6),
    ]
    words = list(word_groups(glyphs))
    assert ["".join(g["decoded"] for g in word) for word in words] == [
        "कुमार",
        "देवी",
        "पिता",
    ]
    assert sorted(g["ordinal"] for word in words for g in word) == [0, 1, 2, 3, 5, 6]


def test_reph_after_space_origin_stays_with_preceding_word():
    glyphs = [
        positioned("प", 85, 103.32, 6, 0),
        positioned("स", 91, 103.32, 6.08, 1),
        positioned(REPH, 97.20, 103.32, 0, 2),
        positioned(" ", 97.08, 103.32, 4.56, 3),
        positioned("राम", 101.64, 103.32, 15, 4),
    ]
    words = list(word_groups(glyphs))
    assert [logical_order("".join(g["decoded"] for g in word)) for word in words] == [
        "पर्स",
        "राम",
    ]


def test_columns_with_no_explicit_space_remain_separate():
    glyphs = [positioned("राम", 10, 84, 15, 0), positioned("देवी", 200, 84, 15, 1)]
    assert len(list(word_groups(glyphs))) == 2


def test_render_gate_refuses_unknown_or_unresolved_ordering():
    references = ReferenceFonts.__new__(ReferenceFonts)
    assert not references.verifies("क�", ["outline"])
    assert not references.verifies(f"क{REPH}", ["outline"])
    assert not references.verifies("क", [None])
