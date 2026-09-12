"""Regression tests for Hindi roll events, source identities and denominators."""

import pytest

from model_training.prep_er_data.jk_hindi_rows import (
    card_frames,
    component_number,
    interpret_summary,
    npr_heading,
    page_sequence,
    parse_card,
    parse_serial,
    reconcile,
    validate_inputs,
)


def event(
    kind, number="1", eligible=True, deleted=False, epic="ABC1234567", name="राम"
):
    return {
        "event_key": f"{kind}:{eligible}:{number}",
        "event_type": kind,
        "number": number,
        "id": epic,
        "assembly_eligible": eligible,
        "deleted_stamp": deleted,
        "elector_name": name,
    }


def test_corrections_replace_fields_without_adding_people():
    base = event("base")
    correction = event("correction", name="राम कुमार")
    rows, issues = reconcile([base, correction])
    assert not issues
    assert len(rows) == 1
    assert rows[0]["active"]
    assert rows[0]["elector_name"] == "राम कुमार"
    assert rows[0]["entry_event_key"] == base["event_key"]
    assert rows[0]["change_event_keys"] == [correction["event_key"]]
    assert base["elector_name"] == "राम"


def test_deletion_stamp_and_list_do_not_subtract_twice():
    rows, issues = reconcile(
        [
            event("base", deleted=True),
            event("addition", "2"),
            event("deletion"),
        ]
    )
    assert not issues
    assert len(rows) == 2
    assert sum(row["active"] for row in rows) == 1
    assert rows[0]["change_event_keys"] == ["deletion:True:1"]


def test_repeated_deletion_list_entry_is_flagged_without_another_subtraction():
    first = event("deletion")
    repeated = {**first, "event_key": "deletion:repeat"}
    rows, issues = reconcile([event("base"), first, repeated])
    assert len(rows) == 1 and not rows[0]["active"]
    assert rows[0]["change_event_keys"] == [first["event_key"], repeated["event_key"]]
    assert issues == [
        {"event_key": repeated["event_key"], "reason": "repeated_deletion_event"}
    ]


def test_npr_and_assembly_serials_are_distinct():
    rows, issues = reconcile(
        [
            event("base"),
            event("base", eligible=False),
            event("deletion", eligible=False),
        ]
    )
    assert not issues
    assert len(rows) == 2
    assert rows[0]["active"]
    assert not rows[1]["active"]


@pytest.mark.parametrize(
    "locality",
    ["मोहल्ला एन पी आर सी टी एम॰ कॉलोनी", "गली शीतला मंदिर एंड एन पी आर"],
)
def test_npr_in_a_locality_is_not_an_assembly_voting_restriction(locality):
    assert not npr_heading("अनुभाग:\n" + locality + " - तहसील-कठुआ")


def test_explicit_npr_restriction_survives_wrapped_heading():
    assert npr_heading(
        "घटक. 3 संशोधन सूची (एन पी आर - िवधानसभा चुनाव मɅ\nमतदान करने के पाğ नहीं)"
    )


def test_duplicate_entry_and_unmatched_change_remain_explicit_issues():
    rows, issues = reconcile(
        [event("base"), event("addition"), event("correction", "999")]
    )
    assert len(rows) == 1
    assert [issue["reason"] for issue in issues] == [
        "duplicate_entry_serial",
        "unmatched_change",
    ]


def test_deletion_id_conflict_does_not_delete_a_different_source_identity():
    rows, issues = reconcile([event("base"), event("deletion", epic="XYZ9999999")])
    assert rows[0]["active"]
    assert issues[0]["reason"] == "deletion_id_conflict"


def test_numbered_blank_name_is_retained_and_counted():
    rows, issues = reconcile([event("base", name=None)])
    assert not issues
    assert len(rows) == 1 and rows[0]["active"]
    assert rows[0]["elector_name"] is None


def summary_rows():
    return [
        {"male": male, "female": female, "total": total}
        for male, female, total in [
            (250, 251, 501),
            (16, 19, 35),
            (266, 270, 536),
            (8, 18, 26),
            (8, 18, 26),
            (258, 252, 510),
            (31, 22, 53),
            (31, 22, 53),
        ]
    ]


def test_closing_summary_uses_net_total_and_keeps_corrections_separate():
    result = interpret_summary(summary_rows())
    assert result["arithmetic_valid"]
    assert result["base_total"] == 501
    assert result["addition_total"] == 35
    assert result["deletion_total"] == 26
    assert result["correction_total"] == 53
    assert result["final_total"] == 510


def test_inconsistent_printed_total_is_preserved_and_flagged():
    rows = summary_rows()
    rows[5]["total"] = 511
    result = interpret_summary(rows)
    assert result["format_supported"]
    assert not result["arithmetic_valid"]
    assert result["final_total"] == 511


def test_unknown_summary_layout_is_not_guessed():
    assert not interpret_summary(summary_rows()[:-1])["format_supported"]


def test_truncated_source_is_detected_from_its_printed_page_count():
    pages = [
        {"page": n, "printed_page": n, "printed_page_total": 43} for n in range(3, 13)
    ]
    result = page_sequence(pages, 12)
    assert result["valid"] is False
    assert result["printed_totals"] == [43]


def test_missing_interior_page_is_detected_even_when_final_footer_survives():
    pages = [
        {"page": 3, "printed_page": 3, "printed_page_total": 5},
        {"page": 4, "printed_page": 5, "printed_page_total": 5},
    ]
    result = page_sequence(pages, 4)
    assert result["valid"] is False
    assert result["position_disagreements"] == [{"physical": 4, "printed": 5}]


def test_absent_footer_does_not_establish_a_complete_page_sequence():
    pages = [
        {"page": 3, "printed_page": 3, "printed_page_total": 4},
        {"page": 4, "printed_page": None, "printed_page_total": None},
    ]
    assert page_sequence(pages, 4)["valid"] is None
    assert page_sequence([], 2)["valid"] is None


def test_missing_horizontal_strokes_do_not_drop_numbered_card_candidates():
    rectangles = [(10, 60, 11, 130), (184, 60, 185, 130)]
    assert card_frames(rectangles) == [(10, 60, 185, 130)]


def test_double_border_strokes_do_not_duplicate_electors():
    rectangles = [
        (10, 60, 11, 140),
        (184, 60, 185, 140),
        (185.08, 60, 186.08, 140),
        (10, 60, 186.08, 61),
        (10, 139, 186.08, 140),
    ]
    assert len(card_frames(rectangles)) == 1


def glyph(text, x, y, font="Mangal", ordinal=0):
    return {
        "text": text,
        "decoded": text,
        "font": font,
        "ordinal": ordinal,
        "origin": (x, y),
        "bbox": (x, y - 8, x + len(text) * 5, y + 2),
        "fingerprint": "test",
    }


class RenderPass:
    def verifies(self, text, fingerprints):
        return "�" not in text


@pytest.mark.parametrize(
    ("number", "title"),
    [("1", "परिवर्धन"), ("2", "विलोपन"), ("3", "संशोधन")],
)
def test_damaged_component_word_can_use_verified_number_and_title(
    monkeypatch, number, title
):
    from model_training.prep_er_data import jk_hindi_rows

    line = [
        {"candidate": value, "verified": valid}
        for value, valid in [
            ("घ�क.", False),
            (number, True),
            (title, True),
            ("सूची", True),
        ]
    ]
    monkeypatch.setattr(jk_hindi_rows, "text_lines", lambda *_: [line])
    assert component_number("घ\x1fक. " + number + " सूची", [], None) == number
    line[2]["verified"] = False
    assert component_number("घ\x1fक. " + number + " सूची", [], None) is None


def test_component_number_and_title_must_agree(monkeypatch):
    from model_training.prep_er_data import jk_hindi_rows

    line = [
        {"candidate": value, "verified": True}
        for value in ("घ�क.", "1", "विलोपन", "सूची")
    ]
    monkeypatch.setattr(jk_hindi_rows, "text_lines", lambda *_: [line])
    assert component_number("घ\x1fक. 1 विलोपन सूची", [], None) is None


def test_card_header_serial_is_not_confused_with_house_number():
    native = [
        glyph("मतदाता का नाम:राम", 12, 85),
        glyph("िपता का नाम:कुमार", 12, 104),
        glyph("मकान संख्या 91", 12, 121),
    ]
    other = [
        glyph("514", 15, 68, "TimesNewRoman"),
        glyph("ABC1234567", 70, 68, "TimesNewRoman"),
    ]
    row = parse_card((10, 60, 185, 140), native, other, RenderPass())
    assert row["number"] == "514"
    assert row["house_no"] == "91"
    assert row["id"] == "ABC1234567"
    assert row["elector_name"] == "राम"
    assert row["relative_type"] == "पिता"


def test_unverified_name_remains_candidate_without_discarding_the_row():
    native = [glyph("मतदाता का नाम:र�म", 12, 85)]
    other = [glyph("1", 15, 68, "TimesNewRoman")]
    row = parse_card((10, 60, 185, 140), native, other, RenderPass())
    assert row["number"] == "1"
    assert row["elector_name"] is None
    assert row["name_candidate"] == "र�म"


def test_wrapped_elector_and_relative_names_keep_their_final_tokens():
    native = [
        glyph("मतदाता का नाम:पूजा कुमारी", 12, 85),
        glyph("ललोत्रा", 12, 97),
        glyph("िपता का नाम:जगन", 12, 112),
        glyph("नाथ", 12, 124),
        glyph("मकान संख्या 87", 12, 139),
        glyph("आय 22", 12, 154),
    ]
    row = parse_card((10, 60, 185, 170), native, [], RenderPass())
    assert row["elector_name"] == "पूजा कुमारी ललोत्रा"
    assert row["relative_name"] == "जगन नाथ"
    assert row["house_no"] == "87"


@pytest.mark.parametrize("damaged_line", [0, 1])
def test_every_line_of_a_wrapped_name_must_pass_rendering(damaged_line):
    texts = ["मतदाता का नाम:पूजा कुमारी", "ललोत्रा"]
    texts[damaged_line] += "�"
    native = [glyph(text, 12, 85 + 12 * n) for n, text in enumerate(texts)]
    row = parse_card((10, 60, 185, 140), native, [], RenderPass())
    assert "ललोत्रा" in row["name_candidate"]
    assert row["elector_name"] is None
    assert row["name_issue"] == "unverified_rendering"


def test_unknown_name_label_is_not_appended_to_the_elector_name():
    native = [
        glyph("मतदाता का नाम:राम", 12, 85),
        glyph("संरक्षक का नाम:कुमार", 12, 101),
        glyph("शाह", 12, 113),
    ]
    row = parse_card((10, 60, 185, 140), native, [], RenderPass())
    assert row["elector_name"] == "राम"
    assert "संरक्षक" in row["candidate_cell"]


def test_wrapped_name_prefix_is_not_mistaken_for_the_age_label():
    native = [
        glyph("मतदाता का नाम:राम", 12, 85),
        glyph("आयुष", 12, 97),
        glyph("मकान संख्या 1", 12, 113),
    ]
    row = parse_card((10, 60, 185, 140), native, [], RenderPass())
    assert row["elector_name"] == "राम आयुष"


def test_empty_template_is_not_an_elector():
    assert parse_card((10, 60, 185, 140), [], [], RenderPass()) is None


def test_duplicate_source_content_cannot_enter_two_split_groups(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"same bytes")
    (tmp_path / "b.pdf").write_bytes(b"same bytes")
    with pytest.raises(ValueError, match="Duplicate PDF content"):
        validate_inputs({"development": ["a.pdf"], "validation": ["b.pdf"]}, [tmp_path])


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("1257", "1257"),
        ("1,257", "1257"),
        ("1,23,456", "123456"),
        ("1,2", None),
        ("123,45", None),
        ("1/2", None),
    ],
)
def test_serial_grouping_is_validated_before_normalization(raw, normalized):
    assert parse_serial(raw) == normalized


def test_comma_serial_in_npr_correction_is_preserved_and_normalized():
    native = [glyph("मतदाता का नाम:राम", 12, 85)]
    other = [glyph("1,257", 15, 68, "TimesNewRoman")]
    row = parse_card((10, 60, 185, 140), native, other, RenderPass())
    assert row["number_raw"] == "1,257"
    assert row["number"] == "1257"


def test_renderable_source_placeholder_does_not_become_an_accepted_name():
    native = [glyph("मतदाता का नाम:राम◌", 12, 85)]
    other = [glyph("1", 15, 68, "TimesNewRoman")]
    row = parse_card((10, 60, 185, 140), native, other, RenderPass())
    assert row["number"] == "1"
    assert row["name_candidate"] == "राम◌"
    assert row["elector_name"] is None
    assert row["name_issue"] == "source_placeholder"


def test_failed_pdf_is_reported_as_failure_instead_of_an_empty_success(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from model_training.prep_er_data import jk_hindi_rows

    source = tmp_path / "broken.pdf"
    source.write_bytes(b"broken source retained for inspection")
    monkeypatch.setattr(
        jk_hindi_rows, "_WORKER_REFERENCES", SimpleNamespace(_shape_cache={})
    )

    def fail(path, references):
        raise ValueError("Unreadable PDF object")

    monkeypatch.setattr(jk_hindi_rows, "parse_pdf", fail)
    events, inventory, audit = jk_hindi_rows.parse_job(source)
    assert events == [] and inventory == []
    assert audit["parse_status"] == "error"
    assert not audit["fully_reconciled"]
    assert audit["issues"] == [
        {
            "reason": "parse_error",
            "type": "ValueError",
            "message": "Unreadable PDF object",
        }
    ]
