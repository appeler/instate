"""Reject incomplete or cross-field Nastaleeq name selections."""

import hashlib
from types import SimpleNamespace

import pymupdf
import pytest

from model_training.prep_er_data import jk_urdu_nastaleeq as parser
from model_training.prep_er_data.jk_urdu_nastaleeq import (
    PdfOutlines,
    decode_line,
    field_groups,
    recover_name,
)

BOX = (0, 0, 180, 66)


def glyph(key, x, y, seqno, **extra):
    return {
        "key": key,
        "origin": (x, y),
        "seqno": seqno,
        "issue": None,
        "outline": key,
        "dir": (1.0, 0.0),
        "size": 2048,
        **extra,
    }


@pytest.fixture
def card():
    return [
        glyph("name", 20, 29, 1),
        glyph("colon", 110, 29, 2),
        glyph("label", 120, 29, 3),
        glyph("relative", 20, 46, 4),
        glyph("colon", 115, 46, 5),
    ]


@pytest.fixture
def decoder():
    def decode(glyphs):
        keys = [g[0] for g in glyphs]
        return {("name",): ("احمد",), ("label",): ("نام ووٹر",)}.get(tuple(keys), ())

    return SimpleNamespace(decode=decode)


def test_valid_card_checks_label_and_excludes_the_relative(card, decoder):
    selected, issue = field_groups(BOX, card, "colon")
    assert issue is None
    assert [g["key"] for g in selected["name"]] == ["name"]
    assert recover_name(BOX, card, "colon", decoder) == {
        "candidate": "احمد",
        "issue": None,
    }


def test_dot_outside_seed_is_retained_from_the_complete_operator(card):
    card.insert(1, glyph("dot", 24, 16, 1))
    selected, issue = field_groups(BOX, card, "colon")
    assert issue is None
    assert [g["key"] for g in selected["name"]] == ["name", "dot"]


def test_source_operator_crossing_into_relative_field_is_rejected(card):
    card[3]["seqno"] = 1
    assert field_groups(BOX, card, "colon")[1] == "operator_crosses_field_boundary"


def test_unassigned_continuation_cannot_silently_disappear(card):
    card.append(glyph("continuation", 20, 36, 6))
    assert field_groups(BOX, card, "colon")[1] == "unassigned_name_line_glyph"


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_key"])
def test_missing_or_ambiguous_separator_rejects_the_field(card, change):
    if change == "missing":
        card.pop(1)
    elif change == "duplicate":
        card.append(glyph("colon", 105, 30, 7))
    else:
        card[1]["key"] = "different"
    assert field_groups(BOX, card, "colon")[1] == "missing_or_ambiguous_line_separator"


def test_unknown_label_and_overlapping_image_prevent_name_acceptance(card, decoder):
    assert (
        recover_name(BOX, card, "colon", decoder, [(10, 18, 40, 33)])["issue"]
        == "name_image_overlap"
    )
    card[2]["outline"] = "wrong label"
    assert (
        recover_name(BOX, card, "colon", decoder)["issue"]
        == "unverified_voter_name_label"
    )


def test_an_ambiguous_name_is_never_selected(card):
    decoder = SimpleNamespace(
        decode=lambda gs: ("نام ووٹر",) if gs[0][0] == "label" else ("بانو", "باںو")
    )
    result = recover_name(BOX, card, "colon", decoder)
    assert result["candidate"] is None
    assert result["issue"] == "ambiguous_unicode"
    assert result["alternatives"] == ("بانو", "باںو")


def test_only_explicit_blank_spaces_split_words():
    seen = []

    def decode(gs):
        seen.append([g[0] for g in gs])
        return ("احمد",) if gs[0][0] == "left" else ("محمد",)

    decoder = SimpleNamespace(decode=decode)
    groups = [
        glyph("left", 0, 29, 1),
        glyph("space", 200, 29, 1),
        glyph("right", 400, 29, 1),
    ]
    assert decode_line(groups, decoder) == (("محمد احمد",), None)
    assert seen == [["left"], ["right"]]
    groups[1]["key"] = "visible glyph"
    groups[1]["outline"] = "visible glyph"
    decode_line(groups, decoder)
    assert seen[-1] == ["left", "visible glyph", "right"]


def test_unknown_glyphs_mixed_sizes_and_reversed_word_positions_reject(decoder):
    unknown = glyph(None, 0, 29, 1, issue="missing_outline")
    assert decode_line([unknown], decoder)[1] == "unresolved_physical_glyph"
    groups = [
        glyph("name", 300, 29, 1),
        glyph("space", 200, 29, 1),
        glyph("name", 100, 29, 1),
    ]
    assert decode_line(groups, decoder)[1] == "nonmonotone_word_positions"
    groups[-1]["size"] = 1000
    assert decode_line(groups, decoder)[1] == "mixed_or_invalid_size"


def test_pdf_cache_tracks_each_pages_font_namespace():
    reference = SimpleNamespace(source_outline=lambda *_: None)
    document = SimpleNamespace()
    reader = PdfOutlines(document, reference)
    span = {
        "font": "same",
        "chars": [(0x628, 4, (20, 29), (20, 20, 30, 30))],
        "size": 10,
        "dir": (1, 0),
        "seqno": 1,
    }
    unsupported = SimpleNamespace(
        get_fonts=lambda: [(1, "cff", None, "same")], get_texttrace=lambda: [span]
    )
    missing = SimpleNamespace(get_fonts=list, get_texttrace=lambda: [span])
    assert reader.page(unsupported)[0]["issue"] == "unsupported_font"
    assert reader.page(missing)[0]["issue"] == "ambiguous_or_missing_outline"
    assert len(reader.resolutions) == 2


def test_partition_enrichment_rejects_changed_source_before_opening(tmp_path):
    from model_training.prep_er_data.jk_urdu_nastaleeq import recover_part

    source = tmp_path / "source.pdf"
    source.write_bytes(b"changed source")
    rows = [{"filename": source.name, "source_sha256": "old hash"}]
    with pytest.raises(ValueError, match="identity"):
        recover_part(rows, source, None, "colon")


@pytest.fixture
def relative_card():
    return [
        glyph("own_label", 120, 29, 1),
        glyph("colon", 110, 29, 1),
        glyph("own", 20, 29, 1),
        glyph("relation_label", 120, 46, 2),
        glyph("colon", 115, 46, 2),
        glyph("relative", 20, 46, 2),
        glyph("header", 50, 11, 2, text="ABC1234567"),
        glyph("header", 15, 10, 2, text="202"),
        glyph("control", 120, 60, 2),
        glyph("colon", 110, 60, 2),
        glyph("house", 20, 60, 2),
    ]


def relative_result(card, *, names=("احمد",), images=()):
    from model_training.prep_er_data.jk_urdu_nastaleeq import recover_relative

    labels = SimpleNamespace(
        decode=lambda gs: {
            ("relation_label",): ("باپ",),
            ("control",): ("خانہ",),
        }.get(tuple(g[0] for g in gs), ())
    )
    reference = SimpleNamespace(
        decode=lambda gs: names if tuple(g[0] for g in gs) == ("relative",) else ()
    )
    return recover_relative(
        BOX, card, "colon", reference, labels, ("ABC1234567", 202), images
    )


def test_relative_stream_can_share_an_operator_with_matched_header(relative_card):
    assert relative_result(relative_card) == {
        "candidate": "احمد",
        "issue": None,
        "relationship": "باپ",
    }


@pytest.mark.parametrize("change", ["id", "number", "extra", "position", "nonascii"])
def test_relative_header_must_match_complete_record_identity(relative_card, change):
    if change == "id":
        relative_card[6]["text"] = "ABC1234568"
    elif change == "number":
        relative_card[7]["text"] = "203"
    elif change == "extra":
        relative_card.insert(8, glyph("header", 80, 11, 2, text="EXTRA"))
    elif change == "position":
        relative_card[6]["origin"] = (190, 11)
    else:
        relative_card[6]["text"] = "ا"  # noqa: RUF001
    assert relative_result(relative_card)["candidate"] is None


@pytest.mark.parametrize("change", ["label", "control", "unknown", "crossing"])
def test_relative_requires_labels_and_all_name_glyphs(relative_card, change):
    if change == "label":
        relative_card[3]["outline"] = "wrong_label"
    elif change == "control":
        relative_card[8]["outline"] = "wrong_control"
    elif change == "unknown":
        relative_card.insert(6, glyph(None, 25, 47, 2, issue="missing_outline"))
    else:
        relative_card.insert(6, glyph("extra", 25, 60, 2))
    assert relative_result(relative_card)["candidate"] is None


def test_relative_withholds_ambiguous_native_name_or_image(relative_card):
    assert (
        relative_result(relative_card, names=("بانو", "باںو"))["issue"]
        == "ambiguous_unicode"
    )
    assert (
        relative_result(relative_card, images=[(10, 40, 30, 50)])["issue"]
        == "name_image_overlap"
    )


@pytest.mark.parametrize("names", [(), ("بانو", "باںو"), ("invalid",)])
def test_verified_label_survives_name_decoding_failure(relative_card, names):
    result = relative_result(relative_card, names=names)
    assert result["candidate"] is None
    assert result["issue"] is not None
    assert result["relationship"] == "باپ"


@pytest.mark.parametrize("position", [3, 8])
def test_failed_label_guard_never_supplies_relationship(relative_card, position):
    relative_card[position]["outline"] = "bad label"
    result = relative_result(relative_card)
    assert result["candidate"] is None
    assert result["relationship"] is None


@pytest.mark.parametrize("existing_label", [None, "ماں"])
def test_partition_keeps_label_without_inventing_relative_name(
    relative_card, tmp_path, monkeypatch, existing_label
):
    source = tmp_path / "part.pdf"
    doc = pymupdf.open()
    doc.new_page(width=600, height=800)
    doc.save(source)
    doc.close()
    row = {
        "filename": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "event_key": "part.pdf:1:1",
        "page": 1,
        "bbox": BOX,
        "id": "ABC1234567",
        "number": 202,
        "relative_name": None,
        "relative_candidate": "unresolved source",
        "relative_type": existing_label,
        "relative_issue": "old issue",
        "elector_name": "علی",
        "active": True,
    }
    reference = SimpleNamespace(decode=lambda _: ())
    labels = SimpleNamespace(
        decode=lambda gs: {
            ("relation_label",): ("باپ",),
            ("control",): ("خانہ",),
        }.get(tuple(g[0] for g in gs), ())
    )
    monkeypatch.setattr(
        parser, "PdfOutlines", lambda *_: SimpleNamespace(page=lambda _: relative_card)
    )
    result, _ = parser.recover_relative_part([row], source, reference, labels, "colon")
    assert result[0]["relative_name"] is None
    assert result[0]["relative_type"] == (existing_label or "باپ")
    assert result[0]["relative_issue"] is not None
    for key in set(row) - {"relative_type", "relative_issue"}:
        assert result[0][key] == row[key]
