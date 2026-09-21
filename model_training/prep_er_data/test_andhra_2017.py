"""Contracts for the Andhra Pradesh positioned-text parser."""

from pathlib import Path

import pymupdf
from andhra_2017 import (
    PrintedControl,
    _collapse_source_duplicates,
    _status,
    load_manifest,
    parse_pdf,
    printed_control,
    source_key,
)


def word(x, y, text):
    return (float(x), float(y), float(x + 5), float(y + 5), text, 0, 0, 0)


def test_source_key_reads_human_archive_name():
    key = source_key("Andhra/5-West Godavari_66-Gopalapuram (SC)_53.pdf")

    assert key.district_number == 5
    assert key.district == "West Godavari"
    assert key.assembly_constituency == 66
    assert key.assembly_name == "Gopalapuram (SC)"
    assert key.part_number == 53


def test_source_key_ignores_archive_copy_suffix():
    key = source_key("Andhra/1-Srikakulam_5-Srikakulam_32 (3).pdf")

    assert key.assembly_constituency == 5
    assert key.part_number == 32


def test_printed_control_uses_net_counts_after_serial_range():
    words = [
        word(10, 700, "1"),
        word(20, 700, "1013"),
        word(30, 700, "501"),
        word(40, 700, "511"),
        word(50, 700, "0"),
        word(60, 700, "1012"),
    ]

    assert printed_control(words) == PrintedControl(1, 1013, 501, 511, 0, 1012)


def test_manifest_separates_available_and_unavailable_rolls(tmp_path):
    manifest = tmp_path / "andhra.csv"
    manifest.write_text(
        "district_name,ac_name,polling_station_number,polling_station_name,"
        "polling_station_location,telugu_file_name,eng_file_name\n"
        "1-Srikakulam,4-Pathapatnam,151,Peddakota,School,telugu.pdf,"
        "andhra_pdfs/english/S01A004P151.PDF\n"
        "1-Srikakulam,4-Pathapatnam,152,Other,Hall,telugu.pdf,"
        "Not available / Unable to download\n"
    )

    available, unavailable = load_manifest(manifest)

    assert unavailable == 1
    assert available[(4, 151)].filename == "S01A004P151"
    assert available[(4, 151)].polling_station_address == "School"


def test_unmatched_events_prevent_a_matched_status():
    control = PrintedControl(1, 1, 1, 0, 0, 1)

    assert (
        _status(
            control,
            boxes_match=True,
            active_match=True,
            sex_match=True,
            conflicting_source_duplicates=0,
            duplicate_serials=0,
            unmatched_deletions=1,
            unmatched_corrections=0,
            records_with_name=1,
            parsed_records=1,
        )
        == "unmatched-events"
    )


def test_repeated_text_layer_prefers_last_version_and_reports_conflict():
    common = {
        "number": 246,
        "id": "JYM2310175",
        "father_or_husband_name": "MARIDAYYA BHEEMUNI",
        "relationship": "father",
        "house_no": "11-57",
        "age": 37,
        "sex": "Male",
        "deleted": False,
        "roll_section": "main",
    }
    first = {**common, "elector_name": "BHIMUNI RAMANA"}
    second = {**common, "elector_name": "RAMANA BHEEMUNI"}
    reused_epic = {
        **second,
        "number": 247,
        "elector_name": "ANOTHER PRINTED CARD",
    }

    rows, duplicates, conflicts = _collapse_source_duplicates(
        [first, second, reused_epic]
    )

    assert rows == [second, reused_epic]
    assert duplicates == 1
    assert conflicts == 1


def _insert_card(
    page,
    *,
    top,
    serial,
    epic,
    name,
    relative,
    sex,
    deleted=False,
):
    page.insert_text((25, top), str(serial), fontsize=7)
    page.insert_text((45, top), epic, fontsize=7)
    page.insert_text((15, top + 13), "Elector's Name:", fontsize=7)
    page.insert_text((65, top + 13), name, fontsize=7)
    page.insert_text((15, top + 33), "Father's Name:", fontsize=7)
    page.insert_text((65, top + 33), relative, fontsize=7)
    page.insert_text((15, top + 50), "House No:", fontsize=7)
    page.insert_text((65, top + 50), "1-1", fontsize=7)
    page.insert_text((15, top + 61), "Age:", fontsize=7)
    page.insert_text((35, top + 61), "30", fontsize=7)
    page.insert_text((65, top + 61), "Sex:", fontsize=7)
    page.insert_text((82, top + 61), sex, fontsize=7)
    if deleted:
        page.insert_text((38, top + 27), "D E L E T E D", fontsize=16)


def _fixture_pdf(path: Path) -> None:
    document = pymupdf.open()
    cover = document.new_page(width=595, height=842)
    cover.insert_text((20, 40), "State - Andhra Pradesh", fontsize=10)
    cover.insert_text((20, 60), "Non Photo Electoral Roll - 2017", fontsize=10)
    cover.insert_text((20, 700), "1 2 0 1 0 1", fontsize=10)
    mother = document.new_page(width=595, height=842)
    _insert_card(
        mother,
        top=70,
        serial=1,
        epic="ABC0000001",
        name="OLD DELETED",
        relative="RELATIVE ONE",
        sex="Male",
        deleted=True,
    )
    _insert_card(
        mother,
        top=145,
        serial=2,
        epic="ABC0000002",
        name="OLD ACTIVE",
        relative="RELATIVE TWO",
        sex="Female",
    )
    deletion = document.new_page(width=595, height=842)
    deletion.insert_text((15, 35), "Component - II : DELETIONS LIST", fontsize=9)
    _insert_card(
        deletion,
        top=70,
        serial=1,
        epic="ABC0000001",
        name="OLD DELETED",
        relative="RELATIVE ONE",
        sex="Male",
    )
    correction = document.new_page(width=595, height=842)
    correction.insert_text((15, 35), "Component - III : CORRECTION LIST", fontsize=9)
    _insert_card(
        correction,
        top=70,
        serial=2,
        epic="ABC0000002",
        name="CORRECTED ACTIVE",
        relative="RELATIVE TWO",
        sex="Female",
    )
    document.save(path)
    document.close()


def test_parse_pdf_applies_deletion_and_correction_cards(tmp_path):
    path = tmp_path / "1-Srikakulam_4-Pathapatnam_151.pdf"
    _fixture_pdf(path)

    rows, audit = parse_pdf(path.read_bytes(), path.name)

    assert audit.status == "matched"
    assert audit.parsed_records == 2
    assert audit.active_records == 1
    assert audit.deletion_records == 1
    assert audit.correction_records == 1
    by_number = {row["number"]: row for row in rows}
    assert by_number[1]["deleted"] is True
    assert by_number[2]["elector_name"] == "CORRECTED ACTIVE"
    assert by_number[2]["corrected"] is True
