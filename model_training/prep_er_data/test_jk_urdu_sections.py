"""Checks for outline-based section recognition and withholding unknown headings."""

import pytest

from model_training.prep_er_data.jk_urdu_sections import classify_pages, heading_markers


def heading(*, extra=False, baseline=80):
    physical = [
        (11, 100, baseline),
        (12, 103, baseline),
        (13, 108, baseline + 3),
        (14, 94, baseline),
        (15, 89, baseline),
    ]
    if extra:
        physical.append((16, 40, baseline))
    spans = [
        {"font": "private", "chars": [[0xFFFD, gid, x, y] for gid, x, y in physical]}
    ]
    outlines = {
        ("private", gid): value
        for gid, value in [
            (11, "part1"),
            (12, "part2"),
            (13, "part3"),
            (14, "hyphen"),
            (15, "two"),
            (16, "other"),
        ]
    }
    catalog = {
        "section_word": ["part1", "part2", "part3"],
        "hyphen": "hyphen",
        "digits": {"two": 2},
        "headings": {},
    }
    return spans, outlines, catalog


def test_section_uses_outlines_even_when_all_unicode_is_damaged():
    spans, outlines, catalog = heading()
    markers = heading_markers(spans, outlines, catalog)
    assert len(markers) == 1
    assert markers[0]["component"] == 2
    catalog["headings"][markers[0]["signature"]] = {
        "event_type": "deletion",
        "assembly_eligible": False,
    }
    result = classify_pages(
        [{"page": 3, "header_spans": spans}, {"page": 4, "header_spans": []}],
        {3: outlines, 4: {}},
        catalog,
    )
    assert all(
        p["event_type"] == "deletion" and p["assembly_eligible"] is False
        for p in result
    )


def test_glyph_number_alone_cannot_establish_a_heading():
    spans, outlines, catalog = heading()
    outlines[("private", 15)] = "a_different_outline"
    assert heading_markers(spans, outlines, catalog) == []


def test_unverified_marker_withholds_instead_of_continuing_previous_section():
    _, _, catalog = heading()
    spans = [
        {"font": "unknown", "chars": [[0x06F1, 1, 510, 80], [ord("-"), 2, 516, 80]]}
    ]
    outlines = {("unknown", 1): "two", ("unknown", 2): "hyphen"}
    result = classify_pages(
        [{"page": 3, "header_spans": spans}], {3: outlines}, catalog
    )
    assert result[0]["event_type"] is None
    assert result[0]["assembly_eligible"] is None
    assert result[0]["markers"][0]["label"] is None


def test_damaged_unicode_digits_do_not_create_section_markers():
    _, _, catalog = heading()
    spans = [
        {"font": "unknown", "chars": [[ord("1"), 1, 510, 80], [ord("-"), 2, 516, 80]]}
    ]
    outlines = {("unknown", 1): "an_urdu_letter", ("unknown", 2): "another_letter"}
    assert heading_markers(spans, outlines, catalog) == []


@pytest.mark.parametrize(("baseline", "expected"), [(60, "deletion"), (100, None)])
def test_numbered_metadata_before_verified_heading_is_not_another_section(
    baseline, expected
):
    spans, outlines, catalog = heading()
    marker = heading_markers(spans, outlines, catalog)[0]
    catalog["headings"][marker["signature"]] = {
        "event_type": "deletion",
        "assembly_eligible": True,
    }
    spans.append(
        {
            "font": "unknown",
            "chars": [[0x06F1, 1, 510, baseline], [ord("-"), 2, 516, baseline]],
        }
    )
    outlines.update({("unknown", 1): "two", ("unknown", 2): "hyphen"})
    result = classify_pages(
        [{"page": 3, "header_spans": spans}], {3: outlines}, catalog
    )
    assert result[0]["event_type"] == expected


def test_unknown_heading_stops_previous_section_label():
    known, outlines, catalog = heading()
    marker = heading_markers(known, outlines, catalog)[0]
    catalog["headings"][marker["signature"]] = {
        "event_type": "deletion",
        "assembly_eligible": True,
    }
    unknown, _, _ = heading(extra=True)
    pages = [
        {"page": n, "header_spans": spans}
        for n, spans in [(3, known), (4, unknown), (5, [])]
    ]
    result = classify_pages(pages, {3: outlines, 4: outlines, 5: {}}, catalog)
    assert result[0]["event_type"] == "deletion"
    assert result[1]["issue"] == "unrecognized_component_heading"
    assert all(
        p["event_type"] is None and p["assembly_eligible"] is None for p in result[1:]
    )


def test_conflicting_headings_are_withheld():
    first, outlines, catalog = heading()
    second, _, _ = heading(baseline=120)
    result = classify_pages(
        [{"page": 3, "header_spans": first + second}], {3: outlines}, catalog
    )
    assert result[0]["issue"] == "multiple_component_headings"
    assert result[0]["event_type"] is None


def test_base_voting_restriction_does_not_require_a_supplement_number():
    _, _, catalog = heading()
    catalog["assembly_restriction_patterns"] = [
        {"glyphs": [["not", 0, 0], ["eligible", 10, 0]]}
    ]
    spans = [{"font": "private", "chars": [[0xFFFD, 1, 200, 80], [0xFFFD, 2, 210, 80]]}]
    outlines = {("private", 1): "not", ("private", 2): "eligible"}
    result = classify_pages(
        [{"page": 3, "header_spans": spans}, {"page": 4, "header_spans": []}],
        {3: outlines, 4: {}},
        catalog,
    )
    assert result[0]["event_type"] == "base"
    assert all(p["assembly_eligible"] is False for p in result)
    assert result[0]["explicit_assembly_restriction"]


def test_restriction_preserves_dot_position_when_font_size_changes():
    from model_training.prep_er_data.jk_urdu_sections import assembly_restriction

    _, _, catalog = heading()
    catalog["assembly_restriction_patterns"] = [
        {"glyphs": [["body", 0, 0], ["dot", 3, -2], ["end", 10, 0]]}
    ]
    spans = [
        {
            "font": "private",
            "chars": [
                [0xFFFD, 1, 100, 80],
                [0xFFFD, 2, 101.8, 78.8],
                [0xFFFD, 3, 106, 80],
                [0xFFFD, 4, 103, 63],
            ],
        }
    ]
    outlines = {
        ("private", 1): "body",
        ("private", 2): "dot",
        ("private", 3): "end",
        ("private", 4): "other_line",
    }
    assert assembly_restriction(spans, outlines, catalog)
    spans[0]["chars"][1][3] = 80
    assert not assembly_restriction(spans, outlines, catalog)


def test_interleaved_pdf_drawing_order_preserves_spatial_section_word():
    spans, outlines, catalog = heading(extra=True)
    chars = spans[0]["chars"]
    spans[0]["chars"] = [chars[0], chars[1], chars[-1], *chars[2:-1]]
    markers = heading_markers(spans, outlines, catalog)
    assert len(markers) == 1
    assert markers[0]["component"] == 2


def test_header_cff_ignores_empty_glyphs_and_preserves_coordinate_scale():
    from types import SimpleNamespace

    from model_training.prep_er_data.jk_urdu_sections import cff_fingerprint

    class Outline:
        def draw(self, pen):
            pen.moveTo((0, 0))
            pen.lineTo((100, 200))
            pen.lineTo((200, 0))
            pen.closePath()

    class Empty:
        def draw(self, pen):
            pass

    font = SimpleNamespace(
        charset=[".notdef", "letter", "space"],
        CharStrings={"letter": Outline(), "space": Empty()},
        FontMatrix=[0.001, 0, 0, 0.001, 0, 0],
    )
    original = cff_fingerprint(font, 1)
    assert original is not None
    assert cff_fingerprint(font, 0) is None
    assert cff_fingerprint(font, 2) is None
    font.FontMatrix = [0.002, 0, 0, 0.002, 0, 0]
    assert cff_fingerprint(font, 1) != original


def test_section_hyphen_can_use_a_separate_verified_font():
    spans, outlines, catalog = heading()
    spans[0]["chars"] = [c for c in spans[0]["chars"] if c[1] != 14]
    spans.append({"font": "punctuation", "chars": [[ord("-"), 7, 94, 80]]})
    outlines[("punctuation", 7)] = "hyphen"
    markers = heading_markers(spans, outlines, catalog)
    assert len(markers) == 1
    assert markers[0]["component"] == 2
    outlines[("punctuation", 7)] = "unknown_shape"
    assert heading_markers(spans, outlines, catalog) == []


@pytest.mark.filterwarnings(
    "ignore:builtin type .* has no __module__ attribute:DeprecationWarning"
)
def test_unembedded_base14_font_resolves_renderer_outlines():
    import pymupdf

    from model_training.prep_er_data.jk_urdu_sections import HeaderFonts

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 80), "-", fontname="helv")
        spans = [
            {"font": s["font"], "chars": [[c[0], c[1], *c[2]] for c in s["chars"]]}
            for s in page.get_texttrace()
        ]
        outlines = HeaderFonts(document).page(page, spans)
        assert outlines[("Helvetica", spans[0]["chars"][0][1])] is not None
