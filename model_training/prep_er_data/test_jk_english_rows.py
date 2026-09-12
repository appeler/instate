"""Synthetic English supplement, name-boundary and source-total regressions."""

from types import SimpleNamespace

import pytest

from model_training.prep_er_data.jk_english_rows import (
    closing_summary,
    component_type,
    npr_heading,
    parse_card,
    source_name_labels,
)
from model_training.prep_er_data.jk_hindi_rows import reconcile


def glyphs(lines, bold=False):
    result = []
    for x, y, text in lines:
        for index, char in enumerate(text):
            result.append(
                {
                    "decoded": char,
                    "text": char,
                    "origin": (x + index * 3, y),
                    "bbox": (x + index * 3, y - 6, x + index * 3 + 3, y + 1),
                    "font": "TimesNewRoman,Bold" if bold else "TimesNewRoman",
                    "ordinal": len(result),
                }
            )
    return result


def card(name="ASHA", relative="Father's Name:RAM"):
    return glyphs(
        [
            (5, 8, "1"),
            (60, 8, "ABC1234567"),
            (5, 24, "Electors Name:" + name),
            (5, 40, relative),
            (5, 57, "House No: 84"),
            (5, 72, "Age: 30 Sex: Female"),
        ]
    )


BOX = (0, 0, 190, 80)


def test_serial_is_not_house_number_and_blank_name_is_retained():
    row = parse_card(BOX, card(name=""))
    assert row["number"] == "1"
    assert row["house_no"] == "84"
    assert row["id"] == "ABC1234567"
    assert row["elector_name"] is None
    assert row["name_issue"] == "missing_name"


def test_wrapped_names_stop_at_the_next_field():
    data = card() + glyphs([(5, 31, "KUMARI"), (5, 48, "KUMAR")])
    row = parse_card(BOX, data)
    assert row["elector_name"] == "ASHA KUMARI"
    assert row["relative_name"] == "RAM KUMAR"
    assert row["age"] == "30" and row["sex_candidate"] == "Female"


def test_damaged_continuation_withholds_whole_name():
    row = parse_card(BOX, card() + glyphs([(5, 31, "K�MARI")]))
    assert row["name_candidate"] == "ASHA K�MARI"
    assert row["elector_name"] is None
    assert row["name_issue"] == "unverified_rendering"


@pytest.mark.parametrize("label", ["Self", "Son in Law", "Adopted Son", "Name"])
def test_less_common_relationships_do_not_drop_records(label):
    row = parse_card(BOX, card(relative=label + ":RAM"))
    assert row["relative_type"] == label
    assert row["relative_name"] == "RAM"


def test_deletion_stamp_does_not_contaminate_name_or_double_subtract():
    stamp = glyphs([(20, 26, "DELETED")], bold=True)
    row = parse_card(BOX, card() + stamp)
    assert row["elector_name"] == "ASHA" and row["deleted_stamp"]
    row.update(event_key="base", event_type="base", assembly_eligible=True)
    deletion = {**row, "event_type": "deletion", "event_key": "deletion"}
    inventory, issues = reconcile([row, deletion])
    assert not issues and len(inventory) == 1
    assert not inventory[0]["active"]


def test_page_label_audit_excludes_overprinted_deletion_glyphs():
    stamp = glyphs([(20, 24, "DELETED")], bold=True)
    assert source_name_labels(card() + stamp) == 1


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Component I - Addition List", "addition"),
        ("Component II - Deletion List", "deletion"),
        ("Component III - Correction List", "correction"),
        ("Component I - Deletion List", None),
    ],
)
def test_component_requires_both_number_and_title(heading, expected):
    assert component_type(heading) == expected


def test_npr_requires_explicit_voting_restriction_and_tolerates_missing_paren():
    assert npr_heading("(NPR-NOT ELIGIBLE TO VOTE IN ASSEMBLY ELECTIONS")
    assert npr_heading("(NPR NOT ELIGIBLE TO VOTE IN ASSEMBLY ELECTIONS)")
    assert not npr_heading("Section: NPR COLONY")


def test_closing_summary_preserves_inconsistent_source_arithmetic():
    def word(text, x, y):
        return {"candidate": text, "x": x, "y": y}

    rows = [(50, 39, 89), (0, 0, 0), (50, 39, 89), (0, 3, 3), (0, 3, 3)]
    rows += [(50, 36, 87), (0, 0, 0), (0, 0, 0)]
    lines = [[word("SUMMARY OF ELECTORS (I+II-III)", 0, 0)]]
    lines.extend(
        [word(str(n), x, y) for n, x in zip(row, (410, 460, 510), strict=True)]
        for y, row in enumerate(rows, 1)
    )
    page = SimpleNamespace(number=13, rect=SimpleNamespace(width=595))
    summary = closing_summary(page, lines)
    assert summary["final_total"] == 87
    assert not summary["arithmetic_valid"]
