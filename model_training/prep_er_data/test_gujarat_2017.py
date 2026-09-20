"""Gujarat embedded-text parser contracts."""

import pytest
from gujarat_2017 import (
    PrintedControl,
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
