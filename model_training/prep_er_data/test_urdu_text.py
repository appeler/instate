"""Guard Urdu body/dot reconstruction, ambiguity and exact rendering checks."""

import pytest
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from model_training.prep_er_data.urdu_text import (
    ReferenceFont,
    combine,
    complete_text,
    reverse_substitutions,
)


def test_shared_body_requires_its_own_complete_dot_sequence():
    mapping = reverse_substitutions(
        {"beh": {"ب"}, "teh": {"ت"}},
        [("body", "initial")],
        [("beh", ["body", "dot"]), ("teh", ["body", "dots"])],
        [],
    )
    values = combine(mapping["initial"], mapping["dot"])
    assert {complete_text(value) for value in values} == {"ب"}
    assert all(complete_text(value) is None for value in mapping["initial"])
    assert all(
        complete_text(value) is None
        for value in combine(mapping["dot"], mapping["initial"])
    )


def test_ligature_reassembles_fragments_before_adding_the_next_letter():
    mapping = reverse_substitutions(
        {"beh": {"ب"}, "alef": {"\u0627"}},
        [],
        [("beh", ["body", "dot"])],
        [(("body", "dot", "alef"), "ligature")],
    )
    assert {complete_text(value) for value in mapping["ligature"]} == {"با"}


def test_missing_middle_fragment_cannot_be_accepted():
    assert complete_text((("ب", 0, 3), ("ب", 2, 3))) is None
    assert complete_text((("ب", 1, 2),)) is None
    assert complete_text((("ب", 0, 2),)) is None
    assert complete_text(()) is None


def test_unbounded_ligatures_raise_instead_of_selecting_a_truncated_candidate():
    with pytest.raises(ValueError, match=r"limit|converge"):
        reverse_substitutions({"beh": {"ب"}}, [], [], [(("beh", "beh"), "beh")])


def test_nested_multiple_substitution_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="nested"):
        reverse_substitutions(
            {"beh": {"ب"}},
            [],
            [("beh", ["body", "dot"]), ("body", ["first", "second"])],
            [],
        )


@pytest.fixture
def reference(tmp_path):
    builder = FontBuilder(1000, isTTF=True)
    names = [".notdef", "space", "beh", "teh", "noon", "body", "dot", "dots"]
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({32: "space", 0x628: "beh", 0x62A: "teh", 0x646: "noon"})
    glyphs = {}
    for index, name in enumerate(names):
        pen = TTGlyphPen(None)
        if name not in (".notdef", "space"):
            pen.moveTo((100, 0))
            pen.lineTo((200 + index * 20, 0))
            pen.lineTo((200, 200 + index * 40))
            pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(dict.fromkeys(names, (600, 0)))
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Synthetic Urdu", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    builder.setupMaxp()
    addOpenTypeFeaturesFromString(
        builder.font,
        """
        languagesystem arab dflt;
        feature ccmp {
            sub beh by body dot;
            sub teh by body dots;
            sub noon by body dot;
        } ccmp;
        """,
    )
    path = tmp_path / "reference.ttf"
    builder.save(path)
    return ReferenceFont(path)


def test_font_shaping_preserves_visually_indistinguishable_unicode(reference):
    assert reference.decode(reference.shape("ت")) == ("ت",)
    assert reference.decode(reference.shape("ب")) == ("ب", "ن")
    assert reference.decode(reference.shape("تت")) == ("تت",)


def test_renderer_rejects_missing_dots_unknown_shapes_and_drawing_order(reference):
    shaped = reference.shape("ت")
    assert len(shaped) == 2
    assert reference.decode(shaped[:1]) == ()
    assert reference.decode(shaped[1:]) == ()
    assert reference.decode(reversed(shaped)) == ()
    assert reference.decode((None,)) == ()
    assert reference.decode(("unknown",)) == ()
    assert reference.decode(()) == ()


def test_gsub_candidate_is_rejected_when_its_context_does_not_reshape(reference):
    shaped = reference.shape("ت")
    reference.shape = lambda _: ("different",)
    assert reference.decode(shaped) == ()
