"""Exercise whole-word geometry with fonts generated independently of source PDFs."""

import copy
import math

import numpy as np
import pytest
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from model_training.prep_er_data.urdu_geometry import (
    BOUND,
    CurvePen,
    PositionedReferenceFont,
    contour_distance,
    mark_variant,
    outline_distance,
    read_outline,
    word_bound,
)


def polygon(points):
    pen = TTGlyphPen(None)
    for contour in points:
        pen.moveTo(contour[0])
        for point in contour[1:]:
            pen.lineTo(point)
        pen.closePath()
    return pen.glyph()


@pytest.fixture
def fonts(tmp_path):
    builder = FontBuilder(2048, isTTF=True)
    names = [".notdef", "space", "beh", "teh", "noon", "body", "dot", "dots"]
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({32: "space", 0x628: "beh", 0x62A: "teh", 0x646: "noon"})
    glyphs = {".notdef": polygon([]), "space": polygon([])}
    for i, name in enumerate(names[2:]):
        glyphs[name] = polygon([[(100, 0), (400 + i * 150, 0), (300, 800 + i * 100)]])
    glyphs["dot"] = polygon([[(0, 0), (200, 0), (200, 200), (0, 200)]])
    glyphs["dots"] = polygon(
        [
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            [(400, 0), (600, 0), (600, 200), (400, 200)],
        ]
    )
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(dict.fromkeys(names, (1000, 0)))
    builder.setupHorizontalHeader(ascent=1600, descent=-400)
    builder.setupNameTable({"familyName": "Urdu Geometry Test", "styleName": "Regular"})
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
    return builder.font, PositionedReferenceFont(path)


def shaped(reference, text):
    return [(reference.outlines[gid], x, y) for gid, x, y in reference.shape(text)]


def test_exact_candidates_keep_indistinguishable_letters(fonts):
    _, reference = fonts
    assert reference.decode(shaped(reference, "ت")) == ("ت",)
    assert reference.decode(shaped(reference, "ب")) == ("ب", "ن")
    assert reference.decode(shaped(reference, "تت")) == ("تت",)


def test_positions_are_global_and_missing_or_moved_dots_reject(fonts):
    _, reference = fonts
    glyphs = shaped(reference, "ت")
    assert reference.decode([(g, x + 17000, y - 48000) for g, x, y in glyphs]) == ("ت",)
    changed = list(glyphs)
    outline, x, y = changed[1]
    changed[1] = (outline, x, y + 256)
    assert reference.decode(changed) == ()
    assert reference.decode(glyphs[:1]) == ()
    assert reference.decode(glyphs[1:]) == ()
    assert reference.decode(reversed(glyphs)) == ()
    assert reference.decode([(None, 0, 0)]) == ()
    assert reference.decode([]) == ()


def test_pdf_glyph_numbers_do_not_determine_unicode(fonts):
    font, reference = fonts
    source = copy.deepcopy(font)
    source.setGlyphOrder([".notdef", *reversed(font.getGlyphOrder()[1:])])
    glyphs = []
    for gid, x, y in reference.shape("ت"):
        name = font.getGlyphName(gid)
        source_gid = source.getGlyphID(name)
        assert source_gid != gid
        glyphs.append((reference.source_outline(source, source_gid), x, y))
    assert reference.decode(glyphs) == ("ت",)


def test_resized_marks_preserve_count_shape_and_relative_centers(fonts):
    font, reference = fonts
    source = copy.deepcopy(font)
    original = read_outline(font, font.getGlyphID("dots"))
    source["glyf"]["dots"] = polygon(
        [
            [(10, 10), (190, 10), (190, 190), (10, 190)],
            [(410, 10), (590, 10), (590, 190), (410, 190)],
        ]
    )
    resized = reference.source_outline(source, source.getGlyphID("dots"))
    assert mark_variant(resized, original)
    glyphs = [
        (resized if gid == font.getGlyphID("dots") else reference.outlines[gid], x, y)
        for gid, x, y in reference.shape("ت")
    ]
    assert reference.decode(glyphs) == ("ت",)
    source["glyf"]["dots"] = polygon(
        [
            [(10, 10), (190, 10), (190, 190), (10, 190)],
            [(610, 10), (790, 10), (790, 190), (610, 190)],
        ]
    )
    moved = read_outline(source, source.getGlyphID("dots"))
    assert not mark_variant(moved, original)
    assert math.isinf(outline_distance(moved, original))
    assert not mark_variant(read_outline(font, font.getGlyphID("dot")), original)


def test_contours_need_one_to_one_assignment(fonts):
    font, _ = fonts
    original = read_outline(font, font.getGlyphID("dots"))
    source = copy.deepcopy(font)
    source["glyf"]["dots"] = polygon(
        [
            [(0, 0), (200, 0), (200, 200), (0, 200)],
            [(0, 0), (200, 0), (200, 200), (0, 200)],
        ]
    )
    duplicate = read_outline(source, source.getGlyphID("dots"))
    assert outline_distance(duplicate, original) > BOUND
    assert not mark_variant(duplicate, original)


def test_quadratic_subdivision_does_not_change_recovered_shape(fonts):
    font, _ = fonts
    source = copy.deepcopy(font)
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.qCurveTo((100, 200), (200, 0))
    pen.closePath()
    source["glyf"]["body"] = pen.glyph()
    first = read_outline(source, source.getGlyphID("body"))
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.qCurveTo((50, 100), (100, 100))
    pen.qCurveTo((150, 100), (200, 0))
    pen.closePath()
    source["glyf"]["body"] = pen.glyph()
    second = read_outline(source, source.getGlyphID("body"))
    assert first.key != second.key
    assert outline_distance(first, second) < 3
    assert contour_distance(first.parts[0], second.parts[0]) == contour_distance(
        second.parts[0], first.parts[0]
    )


def test_word_bound_accounts_for_unequal_glyph_errors():
    assert word_bound(np.asarray([[0, 0], [40, 0]]), [0, 30]) == 35
    assert word_bound(np.asarray([[0, 0], [40, 0]]), [0, 20]) == 30
    assert word_bound(np.asarray([[0, 0], [64, 0]]), [0, 0]) == BOUND
    assert word_bound(np.asarray([[0, 0], [65, 0]]), [0, 0]) > BOUND
    assert math.isinf(word_bound(np.asarray([[np.nan, 0]]), [0]))
    assert math.isinf(word_bound(np.empty((0, 2)), []))


def test_unknown_units_notdef_and_unsupported_curves(fonts):
    font, reference = fonts
    assert reference.source_outline(font, 0) is None
    assert reference.source_outline(font, -1) is None
    assert reference.source_outline(font, 10000) is None
    font = copy.deepcopy(font)
    font["head"].unitsPerEm = 1000
    with pytest.raises(ValueError, match="2048"):
        reference.source_outline(font, 1)
    pen = CurvePen()
    pen.moveTo((0, 0))
    with pytest.raises(ValueError, match="Cubic"):
        pen.curveTo((0, 1), (1, 1), (1, 0))
    with pytest.raises(ValueError, match="Open"):
        pen.endPath()


def test_exact_match_does_not_hide_another_nearby_unicode_candidate(fonts, tmp_path):
    font, _ = fonts
    font = copy.deepcopy(font)
    font["glyf"]["dots"] = polygon([[(0, 0), (202, 0), (202, 202), (0, 202)]])
    path = tmp_path / "nearby-reference.ttf"
    font.save(path)
    reference = PositionedReferenceFont(path)
    assert reference.decode(shaped(reference, "ب")) == ("ب", "ت", "ن")


@pytest.mark.parametrize(
    "transform",
    [(1, 0, 0, 1, 40, -40), (-1, 0, 0, 1, 200, 0), (0.5, 0, 0, 0.5, 10, 20)],
)
def test_composite_components_preserve_raw_coordinates_and_winding(fonts, transform):
    font, _ = fonts
    font = copy.deepcopy(font)
    pen = TTGlyphPen(font["glyf"].glyphs)
    pen.addComponent("body", transform)
    font["glyf"]["teh"] = pen.glyph()
    expected = copy.deepcopy(font["glyf"]["body"])
    a, b, c, d, x, y = transform
    expected.coordinates.transform(((a, b), (c, d)))
    expected.coordinates.translate((x, y))
    font["glyf"]["dots"] = expected
    font["hmtx"].metrics["body"] = (65534, 9000)
    actual = read_outline(font, font.getGlyphID("teh"))
    flat = read_outline(font, font.getGlyphID("dots"))
    assert actual.key == flat.key
    assert np.array_equal(actual.center, flat.center)
    assert contour_distance(actual.parts[0], flat.parts[0]) == 0
    nested = TTGlyphPen(font["glyf"].glyphs)
    nested.addComponent("teh", (1, 0, 0, 1, 0, 0))
    font["glyf"]["noon"] = nested.glyph()
    assert read_outline(font, font.getGlyphID("noon")).key == flat.key


def test_integrated_label_cluster_keeps_body_topology_and_relative_dots(
    fonts, tmp_path
):
    from model_training.prep_er_data.urdu_geometry import LabelReferenceFont

    font, _ = fonts
    font = copy.deepcopy(font)
    body = [(0, 0), (1200, 0), (1100, 500), (0, 500)]
    dots = [
        [(100, -400), (300, -400), (300, -200), (100, -200)],
        [(500, -400), (700, -400), (700, -200), (500, -200)],
    ]
    font["glyf"]["beh"] = polygon([body, *dots])
    path = tmp_path / "integrated-reference.ttf"
    font.save(path)
    reference = LabelReferenceFont(path)
    gid = font.getGlyphID("beh")
    original = reference.outlines[gid]
    shrunk = [
        [(round(400 + 0.9 * (x - 400)), round(-300 + 0.9 * (y + 300))) for x, y in dot]
        for dot in dots
    ]
    font["glyf"]["beh"] = polygon([body, *shrunk])
    assert reference.mixed_distance(read_outline(font, gid), original) <= BOUND
    variants = [
        [body, shrunk[0]],
        [body, shrunk[0], [(x + 100, y) for x, y in shrunk[1]]],
        [body, *[[(x, y - 150) for x, y in dot] for dot in shrunk]],
        [[(x + (100 if x else 0), y) for x, y in body], *shrunk],
        [body, list(reversed(shrunk[0])), shrunk[1]],
        [body, *[[(round(400 + 0.7 * (x - 400)), y) for x, y in dot] for dot in dots]],
    ]
    for contours in variants:
        font["glyf"]["beh"] = polygon(contours)
        assert reference.mixed_distance(read_outline(font, gid), original) > BOUND
