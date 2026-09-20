"""Geometry, field-boundary and source-placeholder checks for native Urdu text."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from model_training.prep_er_data.jk_urdu_unicode import (
    DotVariants,
    UnicodeFonts,
    contours_match,
    native_text,
    parse_card,
    verified_span,
)


@pytest.fixture
def dotted_font():
    builder = FontBuilder(2048, isTTF=True)
    builder.setupGlyphOrder([".notdef", "beh", "teh", "other"])
    builder.setupCharacterMap({0x628: "beh", 0x62A: "teh", 0x646: "other"})
    body = [(0, 0), (600, 0), (600, 300), (0, 300)]
    dot = [(250, -200), (300, -150), (350, -200), (300, -250)]
    glyphs = {}
    for name, marks in {
        ".notdef": [],
        "beh": [body, dot],
        "teh": [
            body,
            [(x, y + 800) for x, y in dot],
            [(x + 150, y + 800) for x, y in dot],
        ],
        "other": [body, [(x, y + 800) for x, y in dot]],
    }.items():
        pen = TTGlyphPen(None)
        for points in marks:
            pen.moveTo(points[0])
            for point in points[1:]:
                pen.lineTo(point)
            pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    return builder.font


def move_dot(font, dx=0, dy=0):
    glyph = font["glyf"]["beh"]
    for i in range(4, 8):
        x, y = glyph.coordinates[i]
        glyph.coordinates[i] = (x + dx, y + dy)


def test_dot_variant_requires_unique_meaning_and_retains_source_letter(dotted_font):
    source = deepcopy(dotted_font)
    move_dot(source, dx=3, dy=-4)
    matcher = DotVariants(dotted_font)
    assert matcher.match(source, 1, 0x628)
    assert not matcher.match(source, 1, 0x646)
    assert not matcher.match(source, 1, 0x62A)


@pytest.mark.parametrize("change", ["far_dot", "body", "curve", "units", "missing_dot"])
def test_dot_variant_rejects_changes_beyond_dot_export_rounding(dotted_font, change):
    source = deepcopy(dotted_font)
    move_dot(source, dx=3)
    glyph = source["glyf"]["beh"]
    if change == "far_dot":
        move_dot(source, dy=5)
    elif change == "body":
        glyph.coordinates[0] = (2, 0)
    elif change == "curve":
        glyph.flags[4] = 0
    elif change == "units":
        source["head"].unitsPerEm = 1000
    elif change == "missing_dot":
        glyph.coordinates = glyph.coordinates[:4]
        glyph.flags = glyph.flags[:4]
        glyph.endPtsOfContours = [3]
        glyph.numberOfContours = 1
    assert not DotVariants(dotted_font).match(source, 1, 0x628)


def test_dot_variant_withholds_visually_ambiguous_unicode(dotted_font):
    source = deepcopy(dotted_font)
    move_dot(source, dx=3)
    for table in dotted_font["cmap"].tables:
        table.cmap[0x646] = "beh"
    assert not DotVariants(dotted_font).match(source, 1, 0x628)
    assert not DotVariants(dotted_font).match(source, 1, 0x646)


def test_dot_variant_rejects_blank_and_unknown_glyphs(dotted_font):
    matcher = DotVariants(dotted_font)
    assert not matcher.match(dotted_font, 0, 0x628)
    assert not matcher.match(dotted_font, 40, 0x628)


@pytest.mark.parametrize("ambiguous_subsets", [False, True])
def test_pdf_dot_checks_cache_by_shape_and_unicode(
    dotted_font, monkeypatch, ambiguous_subsets
):
    source = deepcopy(dotted_font)
    move_dot(source, dx=3)
    alternate = deepcopy(dotted_font)
    move_dot(alternate, dx=4)
    fonts = iter([source, alternate])
    monkeypatch.setattr("fontTools.ttLib.TTFont", lambda _: next(fonts))
    document = SimpleNamespace(extract_font=lambda _: (None, None, None, b""))
    page = SimpleNamespace(
        get_fonts=lambda: (
            [(1, "ttf", "TrueType", "ABC+test")]
            + ([(2, "ttf", "TrueType", "DEF+test")] if ambiguous_subsets else [])
        ),
        get_texttrace=lambda: [
            {
                "font": "test",
                "chars": [
                    (cp, 1, (i * 10, 0), (i * 10, 0, i * 10 + 5, 5))
                    for i, cp in enumerate([0x628, 0x646, 0x628])
                ],
            }
        ],
    )
    checker = UnicodeFonts(document, dotted_font)
    result = checker.page(page)
    assert [r["char"] for r in result] == ["ب", "ن", "ب"]
    assert [r["verified"] for r in result] == (
        [False, False, False] if ambiguous_subsets else [True, False, True]
    )
    assert len(checker.dot_matches) == (0 if ambiguous_subsets else 2)
    assert checker.checks["verified_dot_variant"] == (0 if ambiguous_subsets else 2)


def test_contours_keep_dots_curve_flags_and_one_unit_bound():
    body = ((0, 0, 1), (10, 0, 1), (5, 8, 0))
    dot = ((3, 12, 1), (5, 12, 1), (4, 14, 1))
    rounded = tuple((x + 1, y - 1, flag) for x, y, flag in dot)
    assert contours_match([dot, body], [body[1:] + body[:1], rounded])
    assert not contours_match([body], [body, dot])
    assert not contours_match([body, dot], [body, dot, dot])
    assert not contours_match(
        [body, dot], [body, tuple((x, y - 20, f) for x, y, f in dot)]
    )
    assert not contours_match([dot], [tuple((x + 2, y, f) for x, y, f in dot)])
    assert not contours_match([body], [tuple((x, y, 1) for x, y, _ in body)])


@pytest.mark.parametrize("text", ["jjjj محمد", "�", "123", "محمد\u202e", "--"])
def test_placeholder_or_control_text_is_not_a_name(text):
    assert native_text(text) is None


def test_normalizes_presentation_forms_without_changing_words():
    assert native_text("ﷲ  محمد") == "الله محمد"


def span(text, x, y, width):
    return {
        "text": text,
        "font": "test",
        "origin": (x, y),
        "bbox": (x, y - 3, x + width, y + 1),
    }


def evidence(spans):
    return [
        {
            "char": char,
            "font": s["font"],
            "origin": (
                s["bbox"][0]
                + (i + 0.5) * (s["bbox"][2] - s["bbox"][0]) / len(s["text"]),
                s["origin"][1],
            ),
            "verified": True,
        }
        for s in spans
        for i, char in enumerate(s["text"])
    ]


def source_card():
    return [
        span("نام ووٹر", 145, 20, 40),
        span("محمد", 80, 20, 40),
        span("باپ", 145, 40, 20),
        span("احمد", 80, 40, 40),
        span("جنسعورت", 10, 60, 35),
        span("عمر", 75, 61, 12),
        span("86", 60, 64, 10),
        span("خانہ", 140, 61, 20),
        span("1", 120, 64, 10),
    ]


def test_relative_name_excludes_sex_and_house_number_does_not_become_age():
    spans = source_card()
    row = parse_card((0, 0, 200, 80), spans, evidence(spans))
    assert row["elector_name"] == "محمد"
    assert row["relative_name"] == "احمد"
    assert row["relative_type"] == "باپ"
    assert row["house_no"] == "1"
    assert row["age"] == "86"
    assert row["sex_candidate"] == "عورت"


def test_incomplete_character_evidence_and_wrong_outline_withhold_the_name():
    spans = source_card()
    glyphs = evidence(spans)
    own = [g for g in glyphs if g["origin"][1] == 20 and g["origin"][0] < 145]
    glyphs.remove(own[0])
    assert not verified_span(spans[1], glyphs)
    row = parse_card((0, 0, 200, 80), spans, glyphs)
    assert row["elector_name"] is None
    assert row["name_candidate"] == "محمد"
    assert row["name_issue"] == "unverified_unicode_outline"
    glyphs = evidence(spans)
    next(g for g in glyphs if g["origin"][1] == 20 and g["origin"][0] < 145)[
        "verified"
    ] = False
    assert parse_card((0, 0, 200, 80), spans, glyphs)["elector_name"] is None


def test_split_spans_do_not_invent_native_word_boundaries():
    spans = source_card()
    spans.append(span("بٹ", 40, 25, 30))
    row = parse_card((0, 0, 200, 80), spans, evidence(spans))
    assert row["elector_name"] is None
    assert row["name_issue"] == "split_name_span_unverified"
