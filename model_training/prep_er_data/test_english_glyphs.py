"""Synthetic outline checks for the original English roll decoder."""

import pytest
from fontTools.cffLib import PrivateDict
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image, ImageDraw

from model_training.prep_er_data.english_glyphs import (
    ReferenceFonts,
    contours_agree,
)


def draw_shape(pen, triangle=False):
    pen.moveTo((100, 0))
    pen.lineTo((500, 0))
    pen.lineTo((300 if triangle else 500, 700))
    if not triangle:
        pen.lineTo((100, 700))
    pen.closePath()


@pytest.fixture
def reference_path(tmp_path):
    builder = FontBuilder(1000, isTTF=True)
    names = [".notdef", "space", "rectangle", "triangle"]
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({32: "space", 65: "rectangle", 66: "triangle"})
    glyphs = {}
    for name in names:
        pen = TTGlyphPen(None)
        if name not in (".notdef", "space"):
            draw_shape(pen, name == "triangle")
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(
        {name: (600, 0 if name in (".notdef", "space") else 100) for name in names}
    )
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Synthetic", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    builder.setupMaxp()
    path = tmp_path / "reference.ttf"
    builder.save(path)
    return path


def source(triangle=False, empty=False, width=600):
    pen = T2CharStringPen(width, None)
    if not empty:
        draw_shape(pen, triangle)
    return pen.getCharString(private=PrivateDict())


MATRIX = [0.001, 0, 0, 0.001, 0, 0]


def test_character_comes_from_cmap_and_outline_not_subset_glyph_name(reference_path):
    refs = ReferenceFonts([reference_path])
    assert refs.decode(source(), MATRIX, "g20") == ("A", "outline_match")
    assert refs.decode(source(triangle=True), MATRIX, "g20") == (
        "B",
        "outline_match",
    )


def test_notdef_is_rejected_even_if_its_outline_matches(reference_path):
    refs = ReferenceFonts([reference_path])
    assert refs.decode(source(), MATRIX, ".notdef") == ("�", "notdef")


def test_empty_space_requires_reference_advance(reference_path):
    refs = ReferenceFonts([reference_path])
    assert refs.decode(source(empty=True), MATRIX, "anonymous") == (
        " ",
        "outline_match",
    )
    assert refs.decode(source(empty=True, width=0), MATRIX, "anonymous")[0] == "�"
    assert refs.decode(source(empty=True, width=300), MATRIX, "anonymous")[0] == "�"


def test_wrong_advance_is_not_a_match(reference_path):
    refs = ReferenceFonts([reference_path])
    assert refs.decode(source(width=700), MATRIX, "g36")[0] == "�"


def test_reference_unicode_aliases_are_ambiguous(reference_path):
    from fontTools.ttLib import TTFont

    with TTFont(reference_path) as font:
        for table in font["cmap"].tables:
            if table.isUnicode():
                table.cmap[67] = "rectangle"
        font.save(reference_path)
    refs = ReferenceFonts([reference_path])
    assert refs.decode(source(), MATRIX, "A") == ("�", "ambiguous_outline")


def test_nonstandard_matrix_is_not_silently_rescaled(reference_path):
    refs = ReferenceFonts([reference_path])
    assert refs.decode(source(), [0.002, 0, 0, 0.002, 0, 0], "A") == (
        "�",
        "unsupported_font_matrix",
    )


def test_contour_agreement_is_bidirectional_and_preserves_inner_strokes():
    outer = Image.new("1", (100, 100))
    ImageDraw.Draw(outer).rectangle((10, 10, 90, 90), outline=1)
    inner = outer.copy()
    ImageDraw.Draw(inner).rectangle((30, 30, 70, 70), outline=1)
    assert contours_agree(outer, outer)
    assert not contours_agree(outer, inner)
    assert not contours_agree(inner, outer)


def test_embedded_truetype_needs_no_cmap_to_recover_its_outlines(reference_path):
    from fontTools.ttLib import TTFont

    refs = ReferenceFonts([reference_path])
    with TTFont(reference_path) as font:
        gid = font.getGlyphID("triangle")
        del font["cmap"]
        assert refs.decode_truetype(font, gid) == ("B", "outline_match")
        assert refs.decode_truetype(font, 0) == ("�", "notdef")
