"""Source-control checks for unclassified Urdu appearances and failure isolation."""

import hashlib
import warnings
from types import SimpleNamespace

import pytest

from model_training.prep_er_data.jk_urdu_audit import (
    closing_summary,
    inspect_pdf,
    isolated_job,
    latin_glyphs,
    parse_card,
    part_identity,
    printed_part,
)


def glyphs(text, x=5, y=8, font="Times-Roman"):
    return [
        {
            "decoded": char,
            "text": char,
            "origin": (x + i * 4, y),
            "bbox": (x + i * 4, y - 6, x + i * 4 + 4, y + 1),
            "font": font,
            "ordinal": i,
        }
        for i, char in enumerate(text)
    ]


def test_private_font_ascii_is_not_treated_as_latin_text():
    char = (ord("1"), 25, (0, 0), (0, 0, 5, 5))
    spans = [
        {"font": "TT21Ct00", "chars": [char]},
        {"font": "Times-Roman", "chars": [char]},
    ]
    result = latin_glyphs(spans)
    assert len(result) == 1
    assert result[0]["font"] == "Times-Roman"


def test_part_header_excludes_card_serials_and_retains_suffix():
    data = glyphs("27 M", x=50, y=90) + glyphs("103", x=40, y=108)
    assert printed_part(data) == "27M"
    assert printed_part(data, closing=True) is None
    assert printed_part(glyphs("103 M", x=60, y=52), closing=True) == "103M"
    assert printed_part(glyphs("27 28", x=45, y=90)) is None


@pytest.mark.parametrize("other", ["103M", "27"])
def test_closing_part_mismatch_is_not_hidden_by_valid_table_arithmetic(other):
    result = part_identity(
        "UACA017PS0027.pdf",
        [{"page": 3, "part": "27M"}, {"page": 4, "part": "27M"}],
        [{"page": 41, "part": other}],
    )
    assert result["valid"] is False
    assert result["reason"] == "printed_part_identity_mismatch"


def test_archive_filename_difference_is_separate_from_printed_header_conflict():
    body = [{"page": 3, "part": "27M"}, {"page": 4, "part": "27M"}]
    closing = [{"page": 41, "part": "27M"}]
    assert part_identity("UACA017PS0027.pdf", body, closing)["valid"] is True
    renamed = part_identity("UACA017PS0103.pdf", body, closing)
    assert renamed["valid"] is True
    assert renamed["filename_matches_printed_part"] is False
    assert part_identity("UACA017PS0027.pdf", body[:1], closing)["valid"] is None
    assert part_identity("UACA017PS0027.pdf", body, [])["valid"] is None


def test_house_number_does_not_create_a_numbered_appearance():
    assert parse_card((0, 0, 175, 75), glyphs("84", y=55)) is None


def test_retains_id_when_serial_missing_and_does_not_accept_names():
    row = parse_card((0, 0, 175, 75), glyphs("ABC1234567", x=60))
    assert row["id"] == "ABC1234567"
    assert row["number"] is None
    assert row["elector_name"] is None
    assert row["event_type"] is None
    assert row["assembly_eligible"] is None
    assert row["name_issue"] == "urdu_text_not_verified"


def test_stamp_does_not_replace_header_serial():
    data = glyphs("1") + glyphs("ABC1234567", x=60)
    data += glyphs("DELETED", y=25, font="Helvetica-Bold")
    row = parse_card((0, 0, 175, 75), data)
    assert row["number"] == "1"
    assert row["deleted_stamp"]


def test_urdu_closing_columns_are_total_female_male():
    data = glyphs("(I+II-III)", x=300, y=20)
    rows = [
        (756, 342, 414),
        (38, 27, 11),
        (794, 369, 425),
        (10, 6, 4),
        (10, 6, 4),
        (784, 363, 421),
        (0, 0, 0),
        (0, 0, 0),
    ]
    for i, row in enumerate(rows):
        for x, number in zip((30, 110, 190), row, strict=True):
            data.extend(glyphs(str(number), x=x, y=100 + i * 25))
    page = SimpleNamespace(number=40, rect=SimpleNamespace(width=595))
    result = closing_summary(page, data)
    assert result["base_total"] == 756
    assert result["final_total"] == 784
    assert result["arithmetic_valid"]
    assert result["numeric_rows"][0]["male"] == 414
    without_formula = [g for g in data if g["origin"][1] != 20]
    without_formula.extend(glyphs("19-Jan-2018", x=400, y=770))
    assert closing_summary(page, without_formula) is None
    assert closing_summary(page, without_formula, last_page=True)["final_total"] == 784
    assert closing_summary(page, without_formula[:-11], last_page=True) is None
    data.extend(glyphs("9", x=30, y=400) + glyphs("8", x=110, y=400))
    data.extend(glyphs("7", x=190, y=400))
    assert not closing_summary(page, data)["format_supported"]


@pytest.fixture
def pdf_module():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="builtin type .* has no __module__ attribute",
            category=DeprecationWarning,
        )
        import pymupdf
    return pymupdf


@pytest.fixture
def synthetic_pdf(tmp_path, pdf_module):
    pymupdf = pdf_module
    path = tmp_path / "synthetic.pdf"
    with pymupdf.open() as document:
        for _ in range(3):
            document.new_page(width=595, height=842)
        page = document[2]
        for x in (30, 250):
            for edge in (x, x + 175):
                page.draw_rect(pymupdf.Rect(edge, 100, edge + 0.5, 175))
            page.insert_text((x + 5, 112), "1", fontname="tiro", fontsize=8)
            page.insert_text((x + 60, 112), "ABC1234567", fontname="tiro", fontsize=8)
            page.insert_text((x + 5, 140), "SYNTHETIC", fontname="tiro", fontsize=8)
        page.insert_text((470, 810), "3 of 3", fontname="tiro", fontsize=8)
        document.save(path)
    return path


def test_closing_inspection_rejects_a_valid_table_for_another_part(
    tmp_path, pdf_module
):
    path = tmp_path / "UACA017PS0027.pdf"
    with pdf_module.open() as document:
        for _ in range(5):
            document.new_page(width=595, height=842)
        for index in (2, 3):
            document[index].insert_text((50, 90), "27 M", fontname="tiro", fontsize=10)
        page = document[4]
        page.insert_text((60, 52), "103 M", fontname="tiro", fontsize=10)
        page.insert_text((300, 180), "(I+II-III)", fontname="tiro", fontsize=10)
        for row, total in enumerate([2, 1, 3, 1, 1, 2, 0, 0]):
            for x, value in zip((30, 110, 190), (total, 0, total), strict=True):
                page.insert_text(
                    (x, 220 + row * 30), str(value), fontname="tiro", fontsize=10
                )
        document.save(path)
    _, audit = inspect_pdf(path, closing_only=True)
    assert audit["printed_arithmetic_valid"]
    assert audit["printed_final_total"] == 2
    assert not audit["closing_control_usable"]
    assert audit["part_identity"]["valid"] is False
    assert {issue["reason"] for issue in audit["issues"]} == {
        "printed_part_identity_mismatch",
        "source_filename_part_mismatch",
    }


def test_inspection_retains_repeat_appearances_without_inventing_a_ledger(
    synthetic_pdf,
):
    rows, audit = inspect_pdf(synthetic_pdf)
    assert len(rows) == audit["appearances"] == 2
    assert len({r["appearance_key"] for r in rows}) == 2
    assert {r["id"] for r in rows} == {"ABC1234567"}
    assert all(r["elector_name"] is None and r["event_type"] is None for r in rows)
    assert all("SYNTHETIC" in r["raw_cell"] for r in rows)
    assert (
        audit["source_sha256"] == hashlib.sha256(synthetic_pdf.read_bytes()).hexdigest()
    )
    assert audit["printed_final_total"] is None
    assert audit["page_sequence"]["valid"]


def test_isolated_inspection_returns_complete_results(synthetic_pdf):
    rows, audit = isolated_job(synthetic_pdf, timeout=10)
    assert audit["status"] == "inspected"
    assert len(rows) == 2


def test_structural_mode_preserves_count_evidence_without_copying_card_text(
    synthetic_pdf,
    pdf_module,
    monkeypatch,
):
    expected, old_audit = inspect_pdf(synthetic_pdf)

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Structural extraction must not copy damaged card text")

    monkeypatch.setattr(pdf_module.Page, "get_textbox", unexpected_copy)
    actual, audit = inspect_pdf(synthetic_pdf, structure_only=True)
    assert [{**row, "raw_cell": None} for row in expected] == actual
    for key in ("appearances", "deletion_stamps", "page_sequence", "closing_summaries"):
        assert audit[key] == old_audit[key]
    assert not audit["raw_card_text_retained"]


def test_structural_headers_exclude_card_and_footer_text(synthetic_pdf, pdf_module):
    with pdf_module.open(synthetic_pdf) as document:
        document[2].insert_text((400, 80), "SECTION 2", fontname="tiro")
        document.saveIncr()
    rows, audit = isolated_job(synthetic_pdf, timeout=10, structure_only=True)
    assert len(rows) == 2
    assert all(row["raw_cell"] is None for row in rows)
    header = "".join(
        chr(char[0])
        for span in audit["pages"][2]["header_spans"]
        for char in span["chars"]
    )
    assert header == "SECTION 2"


def test_timeout_discards_partial_results(synthetic_pdf):
    rows, audit = isolated_job(synthetic_pdf, timeout=0.001)
    assert rows == []
    assert audit["status"] == "inspection_timeout"
    assert audit["printed_final_total"] is None
    assert (
        audit["source_sha256"] == hashlib.sha256(synthetic_pdf.read_bytes()).hexdigest()
    )


def test_invalid_pdf_is_an_error_not_a_zero_elector_part(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a pdf")
    rows, audit = isolated_job(path, timeout=10)
    assert rows == []
    assert audit["status"] == "inspection_error"
    assert audit["printed_final_total"] is None


def test_closing_only_does_not_report_zero_appearances(synthetic_pdf):
    rows, audit = isolated_job(synthetic_pdf, timeout=10, closing_only=True)
    assert rows == []
    assert audit["status"] == "inspected"
    assert audit["scope"] == "closing_pages_only"
    assert audit["appearances"] is None
    assert audit["deletion_stamps"] is None
    assert audit["physical_pages"] == 3
    assert audit["page_sequence"]["valid"] is None
    assert [p["page"] for p in audit["pages"]] == [2, 3]
    assert "no_numbered_cards" not in [i["reason"] for i in audit["issues"]]


def test_closing_table_can_precede_its_dated_signature(tmp_path, pdf_module):
    pymupdf = pdf_module

    path = tmp_path / "overflow.pdf"
    triples = [
        (756, 342, 414),
        (38, 27, 11),
        (794, 369, 425),
        (10, 6, 4),
        (10, 6, 4),
        (784, 363, 421),
        (0, 0, 0),
        (0, 0, 0),
    ]
    with pymupdf.open() as document:
        for _ in range(4):
            document.new_page(width=595, height=842)
        for i, triple in enumerate(triples):
            for x, value in zip((30, 110, 190), triple, strict=True):
                document[2].insert_text((x, 100 + i * 25), str(value), fontname="tiro")
        document[3].insert_text((400, 770), "19-Jan-2018", fontname="tiro")
        document.save(path)
    rows, audit = inspect_pdf(path, closing_only=True)
    assert rows == []
    assert audit["printed_final_total"] == 784
    assert audit["closing_summaries"][0]["page"] == 3
    assert audit["printed_arithmetic_valid"]


@pytest.mark.parametrize("closing_only", [False, True])
def test_zero_page_pdf_is_a_source_failure_without_last_page_access(
    tmp_path, closing_only
):
    path = tmp_path / "zero-pages.pdf"
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = []
    for obj in objects:
        offsets.append(len(data))
        data.extend(obj)
    start = len(data)
    data.extend(b"xref\n0 3\n0000000000 65535 f \n")
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    path.write_bytes(data)
    rows, audit = isolated_job(path, 5, closing_only=closing_only)
    assert rows == []
    assert audit["status"] == "source_has_no_pages"
    assert audit["physical_pages"] == 0
    assert audit["printed_final_total"] is None
    assert audit["source_sha256"] == hashlib.sha256(data).hexdigest()
