"""Contracts for the hybrid Andhra recovery builder."""

import io

from andhra_2017 import ManifestRow
from build_andhra_recovery import iter_historical_parts, resolve_historical_part


def _row(number, name, *, change="", total="2"):
    return {
        "number": str(number),
        "id": f"ID{number}",
        "elector_name": name,
        "father_or_husband_name": "Rama Rao",
        "has_husband": "no",
        "house_no": "1",
        "age": "30",
        "sex": "Male",
        "ac_name": "1-Test",
        "part_no": "1",
        "year": "2017",
        "filename": "S01A001P001.PDF",
        "district": "1-Test",
        "polling_station_name": "",
        "polling_station_address": "",
        "net_electors_male": "2",
        "net_electors_female": "0",
        "net_electors_third_gender": "0",
        "net_electors_total": total,
        "change": change,
    }


def test_historical_corrections_replace_the_original_card():
    rows = [
        _row(1, "Old Name"),
        _row(2, "Second Rao"),
        _row(1, "Corrected Rao", change="corrected"),
    ]
    metadata = ManifestRow("S01A001P001", "1-Test", "1-Test", "Room", "School")

    records, audit = resolve_historical_part(rows, metadata=metadata)

    assert len(records) == 2
    assert records[0]["elector_name"] == "Corrected Rao"
    assert records[0]["corrected"]
    assert audit.active_records == audit.printed_total == 2
    assert audit.status == "matched"


def test_historical_invalid_number_does_not_shift_change_types():
    invalid = _row("", "Unreadable", change="deleted")
    rows = [invalid, _row(1, "First Rao"), _row(2, "Second Rao")]
    metadata = ManifestRow("S01A001P001", "1-Test", "1-Test", "Room", "School")

    records, audit = resolve_historical_part(rows, metadata=metadata)

    assert [record["number"] for record in records] == [1, 2]
    assert not any(record["deleted"] for record in records)
    assert audit.active_records == audit.printed_total == 2


def test_concatenated_csv_headers_do_not_split_a_part():
    first = _row(1, "First")
    second = _row(2, "Second")
    header = list(first)
    text = io.StringIO()
    text.write(",".join(header) + "\n")
    text.write(",".join(first[key] for key in header) + "\n")
    text.write(",".join(header) + "\n")
    text.write(",".join(second[key] for key in header) + "\n")
    text.seek(0)

    parts = list(iter_historical_parts(text))

    assert len(parts) == 1
    assert [row["number"] for row in parts[0]] == ["1", "2"]
