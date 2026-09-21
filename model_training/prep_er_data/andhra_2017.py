# pyright: reportMissingImports=false
"""Parse Andhra Pradesh's English 2017 electoral-roll PDFs.

The published rolls are searchable Crystal Reports PDFs.  Each part contains a
mother roll, additions, deletions, corrections, and a printed elector control.
This parser reads the positioned text directly, applies correction cards to the
original elector, marks deletion-list identities as deleted, and reconciles the
active rows to the printed sex and total counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

SOURCE_RE = re.compile(
    r"^(?P<district_no>\d+)-(?P<district>.+)_"
    r"(?P<ac_no>\d+)-(?P<ac>.+)_(?P<part>\d+)(?: \(\d+\))?\.pdf$",
    re.IGNORECASE,
)
NAME_LABEL_RE = re.compile(r"^Elector'?s$", re.IGNORECASE)
RELATION_LABEL_RE = re.compile(
    r"^(Father|Husband|Mother|Other|Guardian|Wife)'?s$", re.IGNORECASE
)
EPIC_RE = re.compile(r"^[A-Z]{2,5}[A-Z0-9/-]*\d[A-Z0-9/-]*$")
INTEGER_RE = re.compile(r"^\d+$")
ACTIVE_SECTIONS = {"main", "addition"}

RECORD_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("number", pa.int32()),
        ("id", pa.string()),
        ("elector_name", pa.string()),
        ("father_or_husband_name", pa.string()),
        ("relationship", pa.string()),
        ("house_no", pa.string()),
        ("age", pa.int16()),
        ("sex", pa.string()),
        ("sex_inferred_from_part_control", pa.bool_()),
        ("deleted", pa.bool_()),
        ("corrected", pa.bool_()),
        ("roll_section", pa.string()),
        ("page", pa.int16()),
        ("correction_page", pa.int16()),
        ("ac_name", pa.string()),
        ("part_no", pa.int16()),
        ("year", pa.int16()),
        ("state", pa.string()),
        ("filename", pa.string()),
        ("source_filename", pa.string()),
        ("district", pa.string()),
        ("polling_station_name", pa.string()),
        ("polling_station_address", pa.string()),
        ("net_electors_male", pa.int32()),
        ("net_electors_female", pa.int32()),
        ("net_electors_third_gender", pa.int32()),
        ("net_electors_total", pa.int32()),
    ]
)

PART_SCHEMA = pa.schema(
    [
        ("source_filename", pa.string()),
        ("filename", pa.string()),
        ("assembly_constituency", pa.int16()),
        ("part_number", pa.int16()),
        ("district", pa.string()),
        ("pdf_bytes", pa.int64()),
        ("pdf_sha256", pa.string()),
        ("pages", pa.int16()),
        ("start_serial", pa.int32()),
        ("end_serial", pa.int32()),
        ("printed_male", pa.int32()),
        ("printed_female", pa.int32()),
        ("printed_third_gender", pa.int32()),
        ("printed_total", pa.int32()),
        ("mother_records", pa.int32()),
        ("addition_records", pa.int32()),
        ("deletion_records", pa.int32()),
        ("correction_records", pa.int32()),
        ("parsed_records", pa.int32()),
        ("active_records", pa.int32()),
        ("active_male", pa.int32()),
        ("active_female", pa.int32()),
        ("active_third_gender", pa.int32()),
        ("sex_inferred_records", pa.int32()),
        ("records_with_name", pa.int32()),
        ("records_with_relative", pa.int32()),
        ("records_with_epic", pa.int32()),
        ("records_with_house", pa.int32()),
        ("records_with_valid_age", pa.int32()),
        ("duplicate_source_records", pa.int32()),
        ("conflicting_source_duplicates", pa.int32()),
        ("duplicate_serials", pa.int32()),
        ("unmatched_deletions", pa.int32()),
        ("unmatched_corrections", pa.int32()),
        ("boxes_match_serial_range", pa.bool_()),
        ("active_match_total", pa.bool_()),
        ("sex_match", pa.bool_()),
        ("serial_contiguous", pa.bool_()),
        ("control_difference", pa.int32()),
        ("status", pa.string()),
        ("error", pa.string()),
    ]
)

Word = tuple[float, float, float, float, str, int, int, int]


@dataclass(frozen=True)
class SourceKey:
    """Geography encoded in an archive member name."""

    district_number: int
    district: str
    assembly_constituency: int
    assembly_name: str
    part_number: int


@dataclass(frozen=True)
class PrintedControl:
    """Printed serial range and active elector counts."""

    start_serial: int | None
    end_serial: int | None
    male: int
    female: int
    third_gender: int
    total: int


@dataclass(frozen=True)
class ManifestRow:
    """One available English PDF from the download inventory."""

    filename: str
    district: str
    ac_name: str
    polling_station_name: str
    polling_station_address: str


@dataclass(frozen=True)
class PartAudit:
    """One source PDF's parse and reconciliation measures."""

    source_filename: str
    filename: str
    assembly_constituency: int
    part_number: int
    district: str
    pdf_bytes: int
    pdf_sha256: str
    pages: int
    start_serial: int | None
    end_serial: int | None
    printed_male: int | None
    printed_female: int | None
    printed_third_gender: int | None
    printed_total: int | None
    mother_records: int
    addition_records: int
    deletion_records: int
    correction_records: int
    parsed_records: int
    active_records: int
    active_male: int
    active_female: int
    active_third_gender: int
    sex_inferred_records: int
    records_with_name: int
    records_with_relative: int
    records_with_epic: int
    records_with_house: int
    records_with_valid_age: int
    duplicate_source_records: int
    conflicting_source_duplicates: int
    duplicate_serials: int
    unmatched_deletions: int
    unmatched_corrections: int
    boxes_match_serial_range: bool
    active_match_total: bool
    sex_match: bool
    serial_contiguous: bool
    control_difference: int | None
    status: str
    error: str | None


def source_key(filename: str) -> SourceKey:
    """Return the district, constituency, and part encoded in a PDF filename.

    The archive contains copied filenames such as ``_32 (3).pdf``. The copy suffix
    is storage metadata; the part identity remains AC and printed part number.
    """
    name = Path(filename).name
    match = SOURCE_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected Andhra filename: {filename}")
    return SourceKey(
        district_number=int(match.group("district_no")),
        district=match.group("district").strip(),
        assembly_constituency=int(match.group("ac_no")),
        assembly_name=match.group("ac").strip(),
        part_number=int(match.group("part")),
    )


def load_manifest(path: Path) -> tuple[dict[tuple[int, int], ManifestRow], int]:
    """Load the scraper inventory and return available English rolls by AC and part."""
    available: dict[tuple[int, int], ManifestRow] = {}
    unavailable = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            ac_match = re.match(r"(\d+)-", row["ac_name"])
            if ac_match is None:
                raise ValueError(f"invalid ac_name in manifest: {row['ac_name']}")
            ac = int(ac_match.group(1))
            part = int(row["polling_station_number"])
            pdf = row["eng_file_name"]
            if not pdf.lower().endswith(".pdf"):
                unavailable += 1
                continue
            key = (ac, part)
            if key in available:
                raise ValueError(f"duplicate Andhra manifest key: {key}")
            available[key] = ManifestRow(
                filename=Path(pdf).stem,
                district=row["district_name"],
                ac_name=row["ac_name"],
                polling_station_name=row["polling_station_name"],
                polling_station_address=row["polling_station_location"],
            )
    return available, unavailable


def printed_control(words: Sequence[Word]) -> PrintedControl | None:
    """Find a self-consistent printed active total on a cover or summary page."""
    numbers = [
        (word[0], word[1], int(value))
        for word in words
        if INTEGER_RE.fullmatch(value := word[4].replace(",", "").strip())
    ]
    candidates: list[PrintedControl] = []
    for _x, y, _value in numbers:
        line = sorted(number for number in numbers if abs(number[1] - y) <= 3.0)
        values = [value for _nx, _ny, value in line]
        for index in range(len(values) - 3):
            male, female, third, total = values[index : index + 4]
            if male + female + third != total or total <= 0:
                continue
            start = values[index - 2] if index >= 2 else None
            end = values[index - 1] if index >= 2 else None
            if start is not None and end is not None and not (0 < start <= end):
                start = end = None
            candidates.append(PrintedControl(start, end, male, female, third, total))
    return max(candidates, key=lambda item: item.total, default=None)


def _column(page_width: float, x: float) -> int:
    first_label = page_width * 0.025
    column_step = page_width * 0.3145
    return min(2, max(0, round((x - first_label) / column_step)))


def _column_bounds(page_width: float, column: int) -> tuple[float, float]:
    first_left = page_width * 0.017
    column_step = page_width * 0.3145
    left = first_left + column * column_step
    right = first_left + (column + 1) * column_step - 1 if column < 2 else page_width
    return left, right


def _same_line(word: Word, y: float, tolerance: float = 2.5) -> bool:
    return abs(word[1] - y) <= tolerance


def _label_pair(words: Sequence[Word], first: Word) -> Word | None:
    return min(
        (
            word
            for word in words
            if word[0] >= first[2] - 1
            and word[0] - first[2] < 12
            and _same_line(word, first[1])
            and word[4].startswith("Name")
        ),
        key=lambda word: word[0],
        default=None,
    )


def _field_words(
    words: Sequence[Word],
    *,
    label_end: float,
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> list[Word]:
    selected = [
        word
        for word in words
        if left <= word[0] < right
        and word[0] >= label_end - 1
        and top - 1.5 <= word[1] < bottom - 1.0
        and word[4].replace(" ", "").upper() != "DELETED"
    ]
    return sorted(selected, key=lambda word: (round(word[1], 1), word[0]))


def _join_words(words: Sequence[Word]) -> str:
    return " ".join(word[4].strip() for word in words if word[4].strip()).strip()


def _inline_value(words: Sequence[Word], label: dict[str, object], right: float) -> str:
    y = float(label["top"])
    extracted = _join_words(
        sorted(
            (
                word
                for word in words
                if float(label["x1"]) - 1 <= word[0] < right
                and _same_line(word, y, tolerance=3.0)
            ),
            key=lambda word: word[0],
        )
    )
    return " ".join(value for value in (str(label["glued"]), extracted) if value)


def _page_stamps(
    page: pymupdf.Page, textpage: pymupdf.TextPage | None = None
) -> list[tuple[float, float, float, float]]:
    stamps: list[tuple[float, float, float, float]] = []
    for block in page.get_text("dict", textpage=textpage)["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = re.sub(r"\W", "", span["text"]).upper()
                if text == "DELETED" or (span["size"] >= 12 and "DELETE" in text):
                    stamps.append(tuple(span["bbox"]))
    return stamps


def elector_boxes(
    page: pymupdf.Page,
    page_number: int,
    *,
    words: Sequence[Word] | None = None,
    textpage: pymupdf.TextPage | None = None,
    stamps: Sequence[tuple[float, float, float, float]] | None = None,
) -> list[dict[str, object]]:
    """Read all actual elector cards from one positioned-text page."""
    words = (
        list(words) if words is not None else page.get_text("words", textpage=textpage)
    )
    labels: list[dict[str, object]] = []
    for word in words:
        text = word[4]
        kind = ""
        pair: Word | None = None
        if NAME_LABEL_RE.fullmatch(text):
            kind = "name"
            pair = _label_pair(words, word)
        elif RELATION_LABEL_RE.fullmatch(text):
            kind = "relation"
            pair = _label_pair(words, word)
        elif text == "House":
            pair = min(
                (
                    candidate
                    for candidate in words
                    if candidate[0] >= word[2] - 1
                    and candidate[0] - word[2] < 12
                    and _same_line(candidate, word[1])
                    and candidate[4].startswith("No")
                ),
                key=lambda candidate: candidate[0],
                default=None,
            )
            kind = "house"
        elif text.startswith("Age"):
            kind = "age"
        elif text.startswith("Sex"):
            kind = "sex"
        if kind and (pair is not None or kind in {"age", "sex"}):
            labels.append(
                {
                    "kind": kind,
                    "x0": word[0],
                    "x1": pair[2] if pair is not None else word[2],
                    "top": word[1],
                    "bottom": max(word[3], pair[3]) if pair is not None else word[3],
                    "text": word[4],
                    "glued": (
                        pair[4].partition(":")[2]
                        if pair is not None
                        else word[4].partition(":")[2]
                    ),
                    "column": _column(page.rect.width, word[0]),
                }
            )
    names = [label for label in labels if label["kind"] == "name"]
    stamps = list(stamps) if stamps is not None else _page_stamps(page, textpage)
    rows: list[dict[str, object]] = []

    def nearest_below(
        kind: str, name_label: dict[str, object], within: float
    ) -> dict[str, object] | None:
        candidates = [
            label
            for label in labels
            if label["kind"] == kind
            and label["column"] == name_label["column"]
            and 0 < float(label["top"]) - float(name_label["top"]) < within
        ]
        return min(
            candidates,
            key=lambda label: float(label["top"]) - float(name_label["top"]),
            default=None,
        )

    for name_label in names:
        column = int(name_label["column"])
        left, right = _column_bounds(page.rect.width, column)
        relation = nearest_below("relation", name_label, 45)
        house = nearest_below("house", name_label, 68)
        age = nearest_below("age", name_label, 85)
        sex = nearest_below("sex", name_label, 85)
        if house is None or age is None or sex is None:
            continue
        head = [
            word
            for word in words
            if left <= word[0] < right
            and float(name_label["top"]) - 18 <= word[1]
            and word[1] < float(name_label["top"]) - 2
        ]
        serials = [
            int(word[4].strip())
            for word in head
            if INTEGER_RE.fullmatch(word[4].strip())
        ]
        if not serials:
            continue
        serial = serials[-1]
        epics = []
        for word in head:
            token = re.sub(r"\s+", "", word[4].upper())
            if EPIC_RE.fullmatch(token):
                epics.append(token)
        epic = epics[-1] if epics else ""
        relation_top = float(relation["top"]) if relation else float(house["top"])
        name = " ".join(
            value
            for value in (
                str(name_label["glued"]),
                _join_words(
                    _field_words(
                        words,
                        label_end=float(name_label["x1"]),
                        top=float(name_label["top"]),
                        bottom=relation_top,
                        left=left,
                        right=right,
                    )
                ),
            )
            if value
        )
        relative = ""
        relationship = ""
        if relation is not None:
            relative = " ".join(
                value
                for value in (
                    str(relation["glued"]),
                    _join_words(
                        _field_words(
                            words,
                            label_end=float(relation["x1"]),
                            top=float(relation["top"]),
                            bottom=float(house["top"]),
                            left=left,
                            right=right,
                        )
                    ),
                )
                if value
            )
            relationship = str(relation["text"]).rstrip("'s").rstrip("'").lower()
        house_no = _inline_value(words, house, right)
        age_text = _inline_value(words, age, float(sex["x0"]))
        age_match = re.search(r"\d+", age_text)
        sex_text = _inline_value(words, sex, right)
        sex_match = re.search(
            r"Male|Female|Third\s+Gender", sex_text, flags=re.IGNORECASE
        )
        box_top = float(name_label["top"]) - 16
        box_bottom = float(age["bottom"]) + 4
        deleted = any(
            box_top <= stamp[1] <= box_bottom
            and left <= (stamp[0] + stamp[2]) / 2 < right
            for stamp in stamps
        )
        rows.append(
            {
                "number": serial,
                "id": epic,
                "elector_name": name,
                "father_or_husband_name": relative,
                "relationship": relationship,
                "house_no": house_no,
                "age": int(age_match.group()) if age_match else None,
                "sex": sex_match.group().title() if sex_match else "",
                "sex_inferred_from_part_control": False,
                "deleted": deleted,
                "corrected": False,
                "roll_section": "main",
                "page": page_number,
                "correction_page": None,
            }
        )
    return rows


def _source_identity(row: dict[str, object]) -> tuple[object, object]:
    """Identify one printed card while retaining repeated EPIC IDs."""
    return row["number"], row["id"]


def _collapse_source_duplicates(
    rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], int, int]:
    """Collapse repeated text layers, preferring the last rendered version."""
    content_fields = (
        "number",
        "id",
        "elector_name",
        "father_or_husband_name",
        "relationship",
        "house_no",
        "age",
        "sex",
        "deleted",
        "roll_section",
    )
    selected: dict[tuple[object, object], dict[str, object]] = {}
    duplicates = 0
    conflicts: set[tuple[object, object]] = set()
    for row in rows:
        key = _source_identity(row)
        previous = selected.get(key)
        if previous is not None:
            duplicates += 1
            if any(previous[field] != row[field] for field in content_fields):
                conflicts.add(key)
        selected[key] = row
    return list(selected.values()), duplicates, len(conflicts)


def _status(
    control: PrintedControl | None,
    *,
    boxes_match: bool,
    active_match: bool,
    sex_match: bool,
    conflicting_source_duplicates: int,
    duplicate_serials: int,
    unmatched_deletions: int,
    unmatched_corrections: int,
    records_with_name: int,
    parsed_records: int,
) -> str:
    if control is None:
        return "missing-control"
    if not boxes_match:
        return "box-mismatch"
    if not active_match:
        return "control-mismatch"
    if not sex_match:
        return "sex-mismatch"
    if conflicting_source_duplicates:
        return "source-duplicate-conflict"
    if duplicate_serials:
        return "duplicate-serials"
    if unmatched_deletions or unmatched_corrections:
        return "unmatched-events"
    if records_with_name != parsed_records:
        return "missing-names"
    return "matched"


def parse_pdf(data: bytes, filename: str) -> tuple[list[dict[str, object]], PartAudit]:
    """Parse and reconcile one Andhra Pradesh roll PDF."""
    key = source_key(filename)
    source_name = Path(filename).name
    digest = hashlib.sha256(data).hexdigest()
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            control = printed_control(document[0].get_text("words"))
            if control is None:
                for page in reversed(document):
                    control = printed_control(page.get_text("words"))
                    if control is not None:
                        break
            events: list[dict[str, object]] = []
            section = "main"
            for page in document:
                textpage = page.get_textpage()
                words: list[Word] = page.get_text("words", textpage=textpage)
                text = " ".join(word[4] for word in words)
                if "ADDITIONS LIST" in text:
                    section = "addition"
                elif "DELETIONS LIST" in text:
                    section = "deletion"
                elif "CORRECTION LIST" in text or "CORRECTIONS LIST" in text:
                    section = "correction"
                elif "SUMMARY OF ELECTORS" in text:
                    section = "summary"
                if section == "summary":
                    continue
                deleted_stamps = (
                    _page_stamps(page, textpage)
                    if "DELETED" in re.sub(r"\W", "", text).upper()
                    else []
                )
                for row in elector_boxes(
                    page,
                    page.number + 1,
                    words=words,
                    stamps=deleted_stamps,
                ):
                    row["roll_section"] = section
                    events.append(row)

            current, duplicate_source_records, conflicting_source_duplicates = (
                _collapse_source_duplicates(
                    [row for row in events if row["roll_section"] in ACTIVE_SECTIONS]
                )
            )
            deletions = [row for row in events if row["roll_section"] == "deletion"]
            corrections = [row for row in events if row["roll_section"] == "correction"]
            by_record = {_source_identity(row): row for row in current}
            id_counts = Counter(str(row["id"]) for row in current if row["id"])
            by_id = {
                str(row["id"]): row
                for row in current
                if row["id"] and id_counts[str(row["id"])] == 1
            }
            by_number = {row["number"]: row for row in current}

            def event_target(event: dict[str, object]) -> dict[str, object] | None:
                exact = by_record.get(_source_identity(event))
                if exact is not None:
                    return exact
                if event["id"]:
                    unique_id = by_id.get(str(event["id"]))
                    if unique_id is not None:
                        return unique_id
                return by_number.get(event["number"])

            unmatched_deletions = 0
            for deletion in deletions:
                target = event_target(deletion)
                if target is None:
                    unmatched_deletions += 1
                else:
                    for field in (
                        "elector_name",
                        "father_or_husband_name",
                        "relationship",
                        "house_no",
                        "age",
                        "sex",
                    ):
                        if deletion.get(field) not in {None, ""}:
                            target[field] = deletion[field]
                    target["deleted"] = True
            unmatched_corrections = 0
            corrected_fields = (
                "elector_name",
                "father_or_husband_name",
                "relationship",
                "house_no",
                "age",
                "sex",
            )
            for correction in corrections:
                target = event_target(correction)
                if target is None:
                    unmatched_corrections += 1
                    continue
                for field in corrected_fields:
                    if correction.get(field) not in {None, ""}:
                        target[field] = correction[field]
                target["corrected"] = True
                target["correction_page"] = correction["page"]

            active = [row for row in current if not row["deleted"]]
            unknown_sex = [
                row
                for row in active
                if row["sex"] not in {"Male", "Female", "Third Gender"}
            ]
            if control is not None and unknown_sex:
                observed = Counter(str(row["sex"]) for row in active)
                deficits = {
                    "Male": control.male - observed["Male"],
                    "Female": control.female - observed["Female"],
                    "Third Gender": control.third_gender - observed["Third Gender"],
                }
                positive = [sex for sex, count in deficits.items() if count > 0]
                if (
                    all(count >= 0 for count in deficits.values())
                    and sum(deficits.values()) == len(unknown_sex)
                    and len(positive) == 1
                ):
                    for row in unknown_sex:
                        row["sex"] = positive[0]
                        row["sex_inferred_from_part_control"] = True
            serials = [int(row["number"]) for row in current]
            duplicate_serials = len(serials) - len(set(serials))
            serial_contiguous = bool(serials) and sorted(serials) == list(
                range(min(serials), min(serials) + len(serials))
            )
            mother = sum(row["roll_section"] == "main" for row in current)
            additions = len(current) - mother
            expected_boxes = (
                control.end_serial - control.start_serial + 1
                if control
                and control.start_serial is not None
                and control.end_serial is not None
                else None
            )
            boxes_match = expected_boxes is not None and len(current) == expected_boxes
            active_male = sum(row["sex"] == "Male" for row in active)
            active_female = sum(row["sex"] == "Female" for row in active)
            active_third = sum(row["sex"] == "Third Gender" for row in active)
            active_match = control is not None and len(active) == control.total
            sex_match = bool(
                control
                and active_male == control.male
                and active_female == control.female
                and active_third == control.third_gender
            )
            records_with_name = sum(bool(row["elector_name"]) for row in current)
            status = _status(
                control,
                boxes_match=boxes_match,
                active_match=active_match,
                sex_match=sex_match,
                conflicting_source_duplicates=conflicting_source_duplicates,
                duplicate_serials=duplicate_serials,
                unmatched_deletions=unmatched_deletions,
                unmatched_corrections=unmatched_corrections,
                records_with_name=records_with_name,
                parsed_records=len(current),
            )
            for row in current:
                row.update(
                    {
                        "record_id": (
                            f"andhra-2017:{key.assembly_constituency}:"
                            f"{key.part_number}:{row['number']}"
                        ),
                        "ac_name": f"{key.assembly_constituency}-{key.assembly_name}",
                        "part_no": key.part_number,
                        "year": 2017,
                        "state": "Andhra Pradesh",
                        "filename": Path(source_name).stem,
                        "source_filename": source_name,
                        "district": f"{key.district_number}-{key.district}",
                        "polling_station_name": "",
                        "polling_station_address": "",
                        "net_electors_male": control.male if control else None,
                        "net_electors_female": control.female if control else None,
                        "net_electors_third_gender": (
                            control.third_gender if control else None
                        ),
                        "net_electors_total": control.total if control else None,
                    }
                )
            audit = PartAudit(
                source_filename=source_name,
                filename=Path(source_name).stem,
                assembly_constituency=key.assembly_constituency,
                part_number=key.part_number,
                district=f"{key.district_number}-{key.district}",
                pdf_bytes=len(data),
                pdf_sha256=digest,
                pages=len(document),
                start_serial=control.start_serial if control else None,
                end_serial=control.end_serial if control else None,
                printed_male=control.male if control else None,
                printed_female=control.female if control else None,
                printed_third_gender=control.third_gender if control else None,
                printed_total=control.total if control else None,
                mother_records=mother,
                addition_records=additions,
                deletion_records=len(deletions),
                correction_records=len(corrections),
                parsed_records=len(current),
                active_records=len(active),
                active_male=active_male,
                active_female=active_female,
                active_third_gender=active_third,
                sex_inferred_records=sum(
                    bool(row["sex_inferred_from_part_control"]) for row in current
                ),
                records_with_name=records_with_name,
                records_with_relative=sum(
                    bool(row["father_or_husband_name"]) for row in current
                ),
                records_with_epic=sum(bool(row["id"]) for row in current),
                records_with_house=sum(bool(row["house_no"]) for row in current),
                records_with_valid_age=sum(
                    isinstance(row["age"], int) and 18 <= row["age"] <= 120
                    for row in current
                ),
                duplicate_source_records=duplicate_source_records,
                conflicting_source_duplicates=conflicting_source_duplicates,
                duplicate_serials=duplicate_serials,
                unmatched_deletions=unmatched_deletions,
                unmatched_corrections=unmatched_corrections,
                boxes_match_serial_range=boxes_match,
                active_match_total=active_match,
                sex_match=sex_match,
                serial_contiguous=serial_contiguous,
                control_difference=(len(active) - control.total if control else None),
                status=status,
                error=None,
            )
            return current, audit
    except Exception as error:
        return [], PartAudit(
            source_filename=source_name,
            filename=Path(source_name).stem,
            assembly_constituency=key.assembly_constituency,
            part_number=key.part_number,
            district=f"{key.district_number}-{key.district}",
            pdf_bytes=len(data),
            pdf_sha256=digest,
            pages=0,
            start_serial=None,
            end_serial=None,
            printed_male=None,
            printed_female=None,
            printed_third_gender=None,
            printed_total=None,
            mother_records=0,
            addition_records=0,
            deletion_records=0,
            correction_records=0,
            parsed_records=0,
            active_records=0,
            active_male=0,
            active_female=0,
            active_third_gender=0,
            sex_inferred_records=0,
            records_with_name=0,
            records_with_relative=0,
            records_with_epic=0,
            records_with_house=0,
            records_with_valid_age=0,
            duplicate_source_records=0,
            conflicting_source_duplicates=0,
            duplicate_serials=0,
            unmatched_deletions=0,
            unmatched_corrections=0,
            boxes_match_serial_range=False,
            active_match_total=False,
            sex_match=False,
            serial_contiguous=False,
            control_difference=None,
            status="parse-error",
            error=f"{type(error).__name__}: {error}",
        )


class ConcatenatedFiles(io.RawIOBase):
    """Expose byte-split archive parts as one readable stream."""

    def __init__(self, parts: Sequence[Path]) -> None:
        """Open the first archive part and retain the ordered part list."""
        self.parts = list(parts)
        self.index = 0
        self.handle = self.parts[0].open("rb") if self.parts else None

    def readable(self) -> bool:
        """Return true because tarfile consumes this stream by reading."""
        return True

    def readinto(self, buffer: bytearray) -> int:
        """Fill a buffer across archive-part boundaries."""
        while self.handle is not None:
            count = self.handle.readinto(buffer)
            if count:
                return count
            self.handle.close()
            self.index += 1
            self.handle = (
                self.parts[self.index].open("rb")
                if self.index < len(self.parts)
                else None
            )
        return 0


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


def _parse_member(
    member: tuple[str, bytes],
) -> tuple[list[dict[str, object]], PartAudit]:
    return parse_pdf(member[1], member[0])


def iter_parsed_pdfs(
    stream: BinaryIO, *, limit: int | None, workers: int
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
            if len(pending) >= workers * 3:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _write_table(
    writer: pq.ParquetWriter, rows: Iterable[dict], schema: pa.Schema
) -> None:
    values = list(rows)
    if values:
        writer.write_table(pa.Table.from_pylist(values, schema=schema))


def parse_tar_stream(
    stream: BinaryIO,
    records_path: Path,
    parts_path: Path,
    summary_path: Path,
    *,
    manifest_path: Path | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Parse a streamed archive into elector records and per-part checks."""
    manifest: dict[tuple[int, int], ManifestRow] = {}
    unavailable = 0
    if manifest_path is not None:
        manifest, unavailable = load_manifest(manifest_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    parts_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    statuses: Counter[str] = Counter()
    files = record_count = active_count = printed_count = 0
    seen_keys: set[tuple[int, int]] = set()
    duplicate_source_keys: list[str] = []
    record_buffer: list[dict[str, object]] = []
    part_buffer: list[dict[str, object]] = []
    temporary_records = records_path.with_suffix(records_path.suffix + ".tmp")
    temporary_parts = parts_path.with_suffix(parts_path.suffix + ".tmp")
    with (
        pq.ParquetWriter(
            temporary_records, RECORD_SCHEMA, compression="zstd"
        ) as record_writer,
        pq.ParquetWriter(
            temporary_parts, PART_SCHEMA, compression="zstd"
        ) as part_writer,
    ):
        for rows, audit in iter_parsed_pdfs(stream, limit=limit, workers=workers):
            key = (audit.assembly_constituency, audit.part_number)
            if key in seen_keys:
                duplicate_source_keys.append(audit.source_filename)
            seen_keys.add(key)
            metadata = manifest.get(key)
            if metadata is not None:
                for row in rows:
                    row.update(
                        {
                            "filename": metadata.filename,
                            "ac_name": metadata.ac_name,
                            "district": metadata.district,
                            "polling_station_name": metadata.polling_station_name,
                            "polling_station_address": metadata.polling_station_address,
                        }
                    )
                audit = PartAudit(
                    **{
                        **asdict(audit),
                        "filename": metadata.filename,
                        "district": metadata.district,
                    }
                )
            record_buffer.extend(rows)
            part_buffer.append(asdict(audit))
            if len(record_buffer) >= 100_000:
                _write_table(record_writer, record_buffer, RECORD_SCHEMA)
                record_buffer.clear()
            if len(part_buffer) >= 256:
                _write_table(part_writer, part_buffer, PART_SCHEMA)
                part_buffer.clear()
            files += 1
            record_count += audit.parsed_records
            active_count += audit.active_records
            printed_count += audit.printed_total or 0
            statuses[audit.status] += 1
            if audit.status != "matched" or files % 250 == 0:
                print(  # noqa: T201 - command progress is the public interface
                    json.dumps(
                        {
                            "file": audit.source_filename,
                            "files": files,
                            "active": active_count,
                            "printed": printed_count,
                            "status": audit.status,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        _write_table(record_writer, record_buffer, RECORD_SCHEMA)
        _write_table(part_writer, part_buffer, PART_SCHEMA)
    temporary_records.replace(records_path)
    temporary_parts.replace(parts_path)
    missing_manifest = sorted(set(manifest) - seen_keys) if manifest else []
    extra_archive = sorted(seen_keys - set(manifest)) if manifest else []
    summary: dict[str, object] = {
        "files": files,
        "parsed_records": record_count,
        "active_records": active_count,
        "printed_records": printed_count,
        "difference": active_count - printed_count,
        "statuses": dict(sorted(statuses.items())),
        "manifest": {
            "available_english_pdfs": len(manifest) if manifest else None,
            "unavailable_english_pdfs": unavailable if manifest else None,
            "missing_from_archive": len(missing_manifest),
            "missing_from_archive_examples": [
                list(key) for key in missing_manifest[:25]
            ],
            "not_in_manifest": len(extra_archive),
            "not_in_manifest_examples": [list(key) for key in extra_archive[:25]],
            "duplicate_archive_keys": len(duplicate_source_keys),
            "duplicate_archive_key_examples": duplicate_source_keys[:25],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    """Parse concatenated archive parts or a gzip tar stream from standard input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_parts", type=Path, nargs="*")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.archive_parts:
        missing = [str(path) for path in args.archive_parts if not path.is_file()]
        if missing:
            parser.error(f"archive parts not found: {', '.join(missing)}")
        raw = ConcatenatedFiles(args.archive_parts)
        stream: BinaryIO = io.BufferedReader(raw, buffer_size=1 << 20)
    else:
        stream = sys.stdin.buffer
    summary = parse_tar_stream(
        stream,
        args.records,
        args.parts,
        args.summary,
        manifest_path=args.manifest,
        limit=args.limit,
        workers=args.workers,
    )
    print(  # noqa: T201 - command result is the public interface
        json.dumps(summary, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
