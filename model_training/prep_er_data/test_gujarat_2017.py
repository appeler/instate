"""Gujarat embedded-text parser contracts."""

import csv
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from build_gujarat_recovery import build_gujarat_token_corpus
from gujarat_2017 import (
    PrintedControl,
    _column_starts,
    _relationship,
    _sex,
    printed_control,
    source_key,
)


def word(x, y, text):
    return (float(x), float(y), float(x + 5), float(y + 5), text, 0, 0, 0)


def test_source_key_uses_constituency_and_part():
    assert source_key("Gujrat/NORMAL_AC137N1370213.pdf") == (137, 213)


def test_source_key_rejects_crossed_constituency_codes():
    with pytest.raises(ValueError, match="unexpected Gujarat filename"):
        source_key("NORMAL_AC137N1380213.pdf")


def test_printed_control_ignores_serial_range_on_same_line():
    words = [
        word(10, 700, "1"),
        word(20, 700, "616"),
        word(30, 700, "325"),
        word(40, 700, "291"),
        word(50, 701.5, "0"),
        word(60, 700, "616"),
    ]
    assert printed_control(words) == PrintedControl(325, 291, 0, 616)


def test_native_relationship_labels_cover_roll_variants():
    assert _relationship("િપતાનું નામ") == "father"
    assert _relationship("પિતાનું નામ") == "father"
    assert _relationship("પતિનું નામ") == "husband"
    assert _relationship("માતાનું નામ") == "mother"


def test_sex_handles_broken_shruti_unicode_mapping():
    assert _sex("પુĮષ") == "male"
    assert _sex("Ęી") == "female"


def test_compact_template_uses_its_leftmost_main_name_label():
    words = [word(40.8 + column * 124.2, 95.2, "નામ") for column in range(4)]
    words.extend(word(60.8 + column * 124.2, 106.0, "નામ") for column in range(4))

    assert _column_starts(words) == [40.8, 165.0, 289.2, 413.4]


def test_token_corpus_learns_the_recovered_font_variant(tmp_path):
    records = tmp_path / "records.parquet"
    parallel = tmp_path / "parallel.csv.gz"
    corpus = tmp_path / "gujarati.csv.gz"
    audit = tmp_path / "audit.json"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_filename": "NORMAL_AC001N0010001.pdf",
                    "epic_id": "GJ1",
                    "elector_name_native": "પટ°લ રમા",
                },
                {
                    "source_filename": "NORMAL_AC001N0010001.pdf",
                    "epic_id": "GJ2",
                    "elector_name_native": "પરમાર રીના",
                },
            ]
        ),
        records,
    )
    with gzip.open(parallel, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "elector_name", "elector_name_t13n", "filename", "number"]
        )
        writer.writerow(["GJ1", "પટેલ રમા", "patel rama", "NORMAL_AC001N0010001", "1"])
        writer.writerow(
            ["GJ2", "પરમાર રીના", "parmar reena", "NORMAL_AC001N0010001", "2"]
        )

    result = build_gujarat_token_corpus(records, parallel, corpus, audit)

    with gzip.open(corpus, "rt", encoding="utf-8", newline="") as handle:
        pairs = {row["gujarati"]: row["english"] for row in csv.DictReader(handle)}
    assert pairs["પટ°લ"] == "patel"
    assert pairs["પટેલ"] == "patel"
    assert pairs["પરમાર"] == "parmar"
    assert result["recovered_coverage"]["mapped_rows"] == 2
    assert json.loads(audit.read_text()) == result
