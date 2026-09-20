# pyright: reportMissingImports=false
"""Parse the embedded text layer in Gujarat's 2017 electoral-roll PDFs.

The published rolls use four voter cards per row.  The historical OCR parser
cropped three cards per row, which discarded an entire column before OCR and
also rejected cards whenever Tesseract did not return exactly five lines.
These PDFs already contain a usable Unicode text layer, so this parser reads
that layer directly and reconciles every part to its printed control total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import BinaryIO

FILENAME_RE = re.compile(
    r"NORMAL_AC(?P<ac>\d{3})N(?P=ac)(?P<part>\d{4})\.pdf$", re.IGNORECASE
)
INTEGER_RE = re.compile(r"^\d+$")
EPIC_RE = re.compile(r"^(?:[A-Z]{2,5}\d{5,}|[A-Z]{1,4}(?:/\d+){2,})$")
MAIN_NAME_LABEL = "નામ"

RECORD_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("source_filename", pa.string()),
        ("assembly_constituency", pa.int16()),
        ("part_number", pa.int16()),
        ("source_page", pa.int16()),
        ("serial_number", pa.int32()),
        ("epic_id", pa.string()),
        ("elector_name_native", pa.string()),
        ("relative_name_native", pa.string()),
        ("relationship", pa.string()),
        ("relationship_label_native", pa.string()),
        ("house_number", pa.string()),
        ("age", pa.int16()),
        ("sex", pa.string()),
        ("sex_native", pa.string()),
        ("year", pa.int16()),
        ("state", pa.string()),
    ]
)

PART_SCHEMA = pa.schema(
    [
        ("source_filename", pa.string()),
        ("assembly_constituency", pa.int16()),
        ("part_number", pa.int16()),
        ("pdf_bytes", pa.int64()),
        ("pdf_sha256", pa.string()),
        ("pages", pa.int16()),
        ("printed_male", pa.int32()),
        ("printed_female", pa.int32()),
        ("printed_third_gender", pa.int32()),
        ("printed_total", pa.int32()),
        ("parsed_records", pa.int32()),
        ("records_with_name", pa.int32()),
        ("records_with_relative", pa.int32()),
        ("records_with_epic", pa.int32()),
        ("duplicate_serials", pa.int32()),
        ("control_difference", pa.int32()),
        ("status", pa.string()),
        ("error", pa.string()),
    ]
)


@dataclass(frozen=True)
class PrintedControl:
    """Printed sex counts from the first page."""

    male: int
    female: int
    third_gender: int
    total: int


@dataclass(frozen=True)
class PartAudit:
    """One PDF's source and reconciliation measures."""

    source_filename: str
    assembly_constituency: int
    part_number: int
    pdf_bytes: int
    pdf_sha256: str
    pages: int
    printed_male: int | None
    printed_female: int | None
    printed_third_gender: int | None
    printed_total: int | None
    parsed_records: int
    records_with_name: int
    records_with_relative: int
    records_with_epic: int
    duplicate_serials: int
    control_difference: int | None
    status: str
    error: str | None


Word = tuple[float, float, float, float, str, int, int, int]


def source_key(filename: str) -> tuple[int, int]:
    """Return the assembly constituency and part encoded in a PDF filename."""
    match = FILENAME_RE.search(Path(filename).name)
    if match is None:
        raise ValueError(f"unexpected Gujarat filename: {filename}")
    return int(match.group("ac")), int(match.group("part"))


def printed_control(words: Sequence[Word]) -> PrintedControl | None:
    """Find a self-consistent printed total on the first page.

    The first page prints male, female, third-gender and total values on one
    baseline.  Other numbers on that line are the first and last serials, so
    candidates are validated by ``male + female + third == total``.
    """
    numbers: list[tuple[float, float, int]] = []
    for x0, y0, _x1, _y1, text, *_ in words:
        value = text.replace(",", "").strip()
        if INTEGER_RE.fullmatch(value):
            numbers.append((x0, y0, int(value)))
    candidates: list[PrintedControl] = []
    for _anchor_x, anchor_y, _anchor_value in numbers:
        values = [
            value
            for _x, _y, value in sorted(
                number for number in numbers if abs(number[1] - anchor_y) <= 3.0
            )
        ]
        for index in range(len(values) - 3):
            male, female, third, total = values[index : index + 4]
            if male + female + third == total and total > 0:
                candidates.append(PrintedControl(male, female, third, total))
    return max(candidates, key=lambda control: control.total, default=None)


def _column_starts(words: Sequence[Word]) -> list[float]:
    counts: Counter[float] = Counter()
    for x0, _y0, _x1, _y1, text, *_ in words:
        if text == MAIN_NAME_LABEL:
            counts[round(x0, 1)] += 1
    observed = sorted(counts)
    if not observed:
        return []
    # A short final page can contain only one or two cards.  Shruti's main-name
    # label starts at x=53.3 in the common template and x=40.8 in the compact
    # template; both use four columns 124.2 points apart.  Anchor on the
    # leftmost observed main label and retain the full grid so short pages parse.
    start = observed[0]
    if not 38.0 <= start <= 56.0:
        return []
    return [round(start + index * 124.2, 1) for index in range(4)]


def _line_words(
    words: Sequence[Word],
    *,
    y: float,
    left: float,
    right: float,
    tolerance: float = 1.5,
) -> list[Word]:
    return sorted(
        (
            word
            for word in words
            if left - 0.5 <= word[0] < right and abs(word[1] - y) <= tolerance
        ),
        key=lambda word: word[0],
    )


def _after_colon(words: Sequence[Word]) -> str:
    colon = next((index for index, word in enumerate(words) if word[4] == ":"), None)
    if colon is None:
        return ""
    return " ".join(word[4] for word in words[colon + 1 :]).strip()


def _relationship(label: str) -> str:
    compact = label.replace(" ", "")
    if "માતા" in compact:
        return "mother"
    if "પિતાનું" in compact or "િપતાનું" in compact:
        return "father"
    if "પતિ" in compact or "પિતનું" in compact:
        return "husband"
    return "other" if compact else ""


def _sex(value: str) -> str:
    compact = value.replace(" ", "")
    if "પુ" in compact:
        return "male"
    if compact:
        return "female"
    return ""


def _integer(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def _reconciliation_status(
    control: PrintedControl | None,
    *,
    parsed_records: int,
    pages: int,
    duplicate_serials: int,
) -> str:
    """Classify a part without treating absent source pages as parser loss."""
    if control is None:
        return "missing-control"
    if control.total > pages * 48:
        return "source-incomplete"
    if parsed_records != control.total:
        return "control-mismatch"
    if duplicate_serials:
        return "duplicate-serials"
    return "matched"


def parse_record_page(page: pymupdf.Page, filename: str) -> list[dict[str, object]]:
    """Parse every voter card from one embedded-text page."""
    ac, part = source_key(filename)
    words: list[Word] = page.get_text("words")
    starts = _column_starts(words)
    if not starts:
        return []
    page_rows: list[dict[str, object]] = []
    for column, left in enumerate(starts):
        right = (
            starts[column + 1] - 1.0 if column + 1 < len(starts) else page.rect.width
        )
        labels = sorted(
            (
                word
                for word in words
                if word[4] == MAIN_NAME_LABEL and abs(word[0] - left) <= 1.0
            ),
            key=lambda word: word[1],
        )
        for label in labels:
            name_y = label[1]
            name_line = _line_words(words, y=name_y, left=left, right=right)
            relation_line = _line_words(
                words, y=name_y + 10.9, left=left, right=right, tolerance=2.0
            )
            house_line = _line_words(
                words, y=name_y + 21.0, left=left, right=right, tolerance=2.0
            )
            age_line = _line_words(
                words, y=name_y + 31.9, left=left, right=right, tolerance=2.0
            )
            id_line = _line_words(
                words, y=name_y - 12.9, left=left, right=right, tolerance=2.2
            )

            serial = next(
                (int(word[4]) for word in id_line if INTEGER_RE.fullmatch(word[4])),
                None,
            )
            if serial is None:
                continue
            epic = next((word[4] for word in id_line if EPIC_RE.fullmatch(word[4])), "")
            relation_colon = next(
                (index for index, word in enumerate(relation_line) if word[4] == ":"),
                None,
            )
            relation_label = (
                " ".join(word[4] for word in relation_line[:relation_colon])
                if relation_colon is not None
                else ""
            )
            sex_native = ""
            colons = [index for index, word in enumerate(age_line) if word[4] == ":"]
            if len(colons) >= 2:
                sex_native = " ".join(word[4] for word in age_line[colons[1] + 1 :])
            age_text = (
                " ".join(word[4] for word in age_line[colons[0] + 1 : colons[1]])
                if len(colons) >= 2
                else _after_colon(age_line)
            )
            serial_label = serial if serial is not None else "missing"
            page_rows.append(
                {
                    "record_id": f"gujarat-2017:{Path(filename).stem}:"
                    f"{serial_label}:{page.number + 1}",
                    "source_filename": Path(filename).name,
                    "assembly_constituency": ac,
                    "part_number": part,
                    "source_page": page.number + 1,
                    "serial_number": serial,
                    "epic_id": epic,
                    "elector_name_native": _after_colon(name_line),
                    "relative_name_native": _after_colon(relation_line),
                    "relationship": _relationship(relation_label),
                    "relationship_label_native": relation_label,
                    "house_number": _after_colon(house_line),
                    "age": _integer(age_text),
                    "sex": _sex(sex_native),
                    "sex_native": sex_native,
                    "year": 2017,
                    "state": "Gujarat",
                }
            )
    return sorted(
        page_rows,
        key=lambda row: (
            row["serial_number"] is None,
            row["serial_number"] or 0,
        ),
    )


def parse_pdf(data: bytes, filename: str) -> tuple[list[dict[str, object]], PartAudit]:
    """Parse and reconcile one Gujarat roll PDF."""
    ac, part = source_key(filename)
    digest = hashlib.sha256(data).hexdigest()
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            control = printed_control(document[0].get_text("words"))
            if control is None and len(document) > 1:
                control = printed_control(document[-1].get_text("words"))
            rows = [
                row for page in document for row in parse_record_page(page, filename)
            ]
            serials = [row["serial_number"] for row in rows if row["serial_number"]]
            duplicates = len(serials) - len(set(serials))
            difference = len(rows) - control.total if control is not None else None
            status = _reconciliation_status(
                control,
                parsed_records=len(rows),
                pages=len(document),
                duplicate_serials=duplicates,
            )
            audit = PartAudit(
                source_filename=Path(filename).name,
                assembly_constituency=ac,
                part_number=part,
                pdf_bytes=len(data),
                pdf_sha256=digest,
                pages=len(document),
                printed_male=control.male if control else None,
                printed_female=control.female if control else None,
                printed_third_gender=control.third_gender if control else None,
                printed_total=control.total if control else None,
                parsed_records=len(rows),
                records_with_name=sum(bool(row["elector_name_native"]) for row in rows),
                records_with_relative=sum(
                    bool(row["relative_name_native"]) for row in rows
                ),
                records_with_epic=sum(bool(row["epic_id"]) for row in rows),
                duplicate_serials=duplicates,
                control_difference=difference,
                status=status,
                error=None,
            )
            return rows, audit
    except Exception as error:
        return [], PartAudit(
            source_filename=Path(filename).name,
            assembly_constituency=ac,
            part_number=part,
            pdf_bytes=len(data),
            pdf_sha256=digest,
            pages=0,
            printed_male=None,
            printed_female=None,
            printed_third_gender=None,
            printed_total=None,
            parsed_records=0,
            records_with_name=0,
            records_with_relative=0,
            records_with_epic=0,
            duplicate_serials=0,
            control_difference=None,
            status="parse-error",
            error=f"{type(error).__name__}: {error}",
        )


def iter_tar_pdfs(
    stream: BinaryIO, limit: int | None = None
) -> Iterator[tuple[str, bytes]]:
    """Yield PDFs from a gzip tar stream without extracting the archive."""
    seen = 0
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            if not data.startswith(b"%PDF"):
                raise ValueError(f"archive member is not a PDF: {member.name}")
            yield member.name, data
            seen += 1
            if limit is not None and seen >= limit:
                return


def _write_table(
    writer: pq.ParquetWriter, rows: Iterable[dict], schema: pa.Schema
) -> None:
    values = list(rows)
    if values:
        writer.write_table(pa.Table.from_pylist(values, schema=schema))


def _parse_member(
    member: tuple[str, bytes],
) -> tuple[list[dict[str, object]], PartAudit]:
    """Parse one archive member in a worker process."""
    return parse_pdf(member[1], member[0])


def iter_parsed_pdfs(
    stream: BinaryIO,
    *,
    limit: int | None,
    workers: int,
) -> Iterator[tuple[list[dict[str, object]], PartAudit]]:
    """Parse PDFs in archive order with bounded process-level concurrency."""
    members = iter_tar_pdfs(stream, limit=limit)
    if workers == 1:
        for filename, data in members:
            yield parse_pdf(data, filename)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for member in members:
            pending.append(executor.submit(_parse_member, member))
            if len(pending) >= workers * 2:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def parse_tar_stream(
    stream: BinaryIO,
    records_path: Path,
    parts_path: Path,
    summary_path: Path,
    *,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Parse a streamed archive into compact records and part audits."""
    records_path.parent.mkdir(parents=True, exist_ok=True)
    parts_path.parent.mkdir(parents=True, exist_ok=True)
    statuses: Counter[str] = Counter()
    records = 0
    printed = 0
    files = 0
    record_buffer: list[dict[str, object]] = []
    part_buffer: list[dict] = []
    with (
        pq.ParquetWriter(
            records_path, RECORD_SCHEMA, compression="zstd"
        ) as record_writer,
        pq.ParquetWriter(parts_path, PART_SCHEMA, compression="zstd") as part_writer,
    ):
        for rows, audit in iter_parsed_pdfs(stream, limit=limit, workers=workers):
            record_buffer.extend(rows)
            part_buffer.append(asdict(audit))
            if len(record_buffer) >= 100_000:
                _write_table(record_writer, record_buffer, RECORD_SCHEMA)
                record_buffer.clear()
            if len(part_buffer) >= 256:
                _write_table(part_writer, part_buffer, PART_SCHEMA)
                part_buffer.clear()
            files += 1
            records += len(rows)
            printed += audit.printed_total or 0
            statuses[audit.status] += 1
            if audit.status != "matched" or files % 100 == 0:
                print(  # noqa: T201 - command progress is the public interface
                    json.dumps(
                        {
                            "file": audit.source_filename,
                            "files": files,
                            "parsed": records,
                            "printed": printed,
                            "status": audit.status,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        _write_table(record_writer, record_buffer, RECORD_SCHEMA)
        _write_table(part_writer, part_buffer, PART_SCHEMA)
    summary: dict[str, object] = {
        "files": files,
        "parsed_records": records,
        "printed_records": printed,
        "difference": records - printed,
        "statuses": dict(sorted(statuses.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    """Parse a concatenated gzip tar stream from standard input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    summary = parse_tar_stream(
        sys.stdin.buffer,
        args.records,
        args.parts,
        args.summary,
        limit=args.limit,
        workers=args.workers,
    )
    print(  # noqa: T201 - command result is the public interface
        json.dumps(summary, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
