"""Numeric recovery must follow outlines and reject incomplete source fields."""

import io

import pyarrow.parquet as pq
import pymupdf
import pytest
from fontTools.cffLib import CFFFontSet
from fontTools.ttLib import TTFont, TTLibError

from model_training.prep_er_data.jk_urdu_demographics import (
    NumericReference,
    bottom_filter,
    recover_card,
    recover_part,
    sha256,
)
from model_training.prep_er_data.jk_urdu_nastaleeq import IndexedPage


@pytest.fixture(scope="module")
def numeric_reference():
    return NumericReference()


def test_numeric_decoder_uses_drawing_not_reported_glyph_name(numeric_reference):
    cff = CFFFontSet()
    cff.decompile(io.BytesIO(pymupdf.Font("Times-Roman").buffer), TTFont())
    font = cff[0]
    value, status = numeric_reference.decode(
        font.CharStrings["three"], font.FontMatrix, "nine"
    )
    assert (value, status) == ("3", "outline_match")
    value, _ = numeric_reference.decode(font.CharStrings["A"], font.FontMatrix, "three")
    assert value == "�"


def test_missing_labels_never_assign_digits_by_position():
    value, issue = recover_card(
        [0, 0, 180, 66],
        IndexedPage([]),
        [{"origin": (80, 60), "text": "3"}],
        "colon",
        None,
        None,
    )
    assert all(item is None for item in value.values())
    assert issue == "missing_label"


def test_numeric_fields_abstain_on_native_letters_or_image_overlap(monkeypatch):
    labels = [
        {"text": "جنس", "origin": (40, 60)},
        {"text": "عمر", "origin": (90, 60)},
        {"text": "خانہ", "origin": (140, 60)},
    ]
    values = []
    for label in labels:
        x, y = label["origin"]
        values.extend(
            [
                {"key": "label", "origin": (x + 5, y), "name": label["text"]},
                {"key": "colon", "origin": (x, y)},
            ]
        )
    values.append({"key": "letter", "origin": (100, 60)})
    glyphs = IndexedPage(values)
    monkeypatch.setattr(
        "model_training.prep_er_data.jk_urdu_demographics._label_before",
        lambda gs, card, sep, allowed, ref: {
            "start": sep["index"] - 1,
            "end": sep["index"],
            "text": gs[sep["index"] - 1]["name"],
        },
    )
    monkeypatch.setattr(
        "model_training.prep_er_data.jk_urdu_demographics.decode_line",
        lambda selected, ref: ((), "unresolved"),
    )
    numbers = [{"origin": (120, 60), "text": "3"}]
    value, _ = recover_card([0, 0, 180, 66], glyphs, numbers, "colon", None, None)
    assert value["house_no"] is None
    glyphs = IndexedPage(values[:-1])
    value, _ = recover_card(
        [0, 0, 180, 66], glyphs, numbers, "colon", None, None, [(110, 50, 130, 64)]
    )
    assert value["house_no"] is None


def test_bottom_band_includes_displaced_marks_but_not_next_card():
    include = bottom_filter([[0, 0, 180, 82], [0, 83, 180, 165]])
    assert include((100, 40))
    assert include((100, 150))
    assert not include((100, 90))


def test_malformed_embedded_font_preserves_identity_and_abstains(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(source)

    def malformed(*args):
        raise TTLibError("truncated hmtx")

    monkeypatch.setattr(
        "model_training.prep_er_data.jk_urdu_nastaleeq.PdfOutlines.page", malformed
    )
    row = {
        "filename": source.name,
        "source_sha256": sha256(source),
        "entry_event_key": "original:1:1",
        "event_key": "original:1:1",
        "number": "1",
        "id": "ABC1234567",
        "page": 1,
        "active": True,
        "assembly_eligible": True,
        "bbox": [0.0, 0.0, 180.0, 66.0],
    }
    report = recover_part(([row], tmp_path, tmp_path))
    result = pq.read_table(tmp_path / "source.pdf.parquet").to_pylist()
    assert len(result) == report["rows"] == 1
    assert all(result[0][key] == value for key, value in row.items())
    assert result[0]["demographic_issue"] == "malformed_embedded_font"
    assert all(result[0][key] is None for key in ("house_no", "age", "sex_candidate"))
