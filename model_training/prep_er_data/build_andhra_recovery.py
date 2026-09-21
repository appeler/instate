# pyright: reportMissingImports=false
"""Combine source-PDF reparses with the remaining historical Andhra rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from andhra_2017 import (
    PART_SCHEMA,
    RECORD_SCHEMA,
    ManifestRow,
    PartAudit,
    _collapse_source_duplicates,
    _source_identity,
    _write_table,
    load_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import TextIO

CANONICAL_FILENAME_RE = re.compile(
    r"^S\d+A(?P<ac>\d{3})P(?P<part>\d{3})\.PDF$", re.IGNORECASE
)
FINAL_RECORD_SCHEMA = RECORD_SCHEMA.append(pa.field("record_source", pa.string()))
FINAL_PART_SCHEMA = PART_SCHEMA.append(pa.field("record_source", pa.string()))


def _integer(value: object) -> int | None:
    text = str(value or "").strip()
    try:
        return int(float(text)) if text else None
    except ValueError:
        return None


def _canonical_key(filename: str) -> tuple[int, int]:
    match = CANONICAL_FILENAME_RE.fullmatch(filename.strip())
    if match is None:
        raise ValueError(f"unexpected canonical Andhra filename: {filename}")
    return int(match.group("ac")), int(match.group("part"))


def _historical_row(
    source: dict[str, str],
    *,
    metadata: ManifestRow | None,
) -> dict[str, object] | None:
    number = _integer(source.get("number"))
    if number is None:
        return None
    filename = source["filename"].strip().upper()
    ac_number, part_number = _canonical_key(filename)
    relative = source.get("father_or_husband_name", "").strip()
    has_husband = source.get("has_husband", "").strip().casefold()
    relationship = "husband" if has_husband in {"1", "true", "yes"} else "father"
    if not relative:
        relationship = ""
    change = source.get("change", "").strip().casefold()
    sex = source.get("sex", "").strip().title()
    if sex in {"Third", "Thirdgender"}:
        sex = "Third Gender"
    age = _integer(source.get("age"))
    if age is not None and not 18 <= age <= 120:
        age = None
    return {
        "record_id": f"andhra-2017:{ac_number}:{part_number}:{number}",
        "number": number,
        "id": source.get("id", "").strip(),
        "elector_name": source.get("elector_name", "").strip(),
        "father_or_husband_name": relative,
        "relationship": relationship,
        "house_no": source.get("house_no", "").strip(),
        "age": age,
        "sex": sex,
        "sex_inferred_from_part_control": False,
        "deleted": change == "deleted",
        "corrected": change == "corrected",
        "roll_section": "addition" if change == "added" else "main",
        "page": None,
        "correction_page": None,
        "ac_name": metadata.ac_name if metadata else source.get("ac_name", "").strip(),
        "part_no": part_number,
        "year": _integer(source.get("year")) or 2017,
        "state": "Andhra Pradesh",
        "filename": metadata.filename if metadata else filename.removesuffix(".PDF"),
        "source_filename": filename,
        "district": metadata.district
        if metadata
        else source.get("district", "").strip(),
        "polling_station_name": (
            metadata.polling_station_name
            if metadata
            else source.get("polling_station_name", "").strip()
        ),
        "polling_station_address": (
            metadata.polling_station_address
            if metadata
            else source.get("polling_station_address", "").strip()
        ),
        "net_electors_male": _integer(source.get("net_electors_male")),
        "net_electors_female": _integer(source.get("net_electors_female")),
        "net_electors_third_gender": _integer(source.get("net_electors_third_gender")),
        "net_electors_total": _integer(source.get("net_electors_total")),
    }


def _event_target(
    event: dict[str, object], current: Sequence[dict[str, object]]
) -> dict[str, object] | None:
    exact = {(_source_identity(row)): row for row in current}.get(
        _source_identity(event)
    )
    if exact is not None:
        return exact
    epic = str(event["id"])
    if epic:
        matches = [row for row in current if row["id"] == epic]
        if len(matches) == 1:
            return matches[0]
    matches = [row for row in current if row["number"] == event["number"]]
    return matches[0] if len(matches) == 1 else None


def resolve_historical_part(
    source_rows: Sequence[dict[str, str]],
    *,
    metadata: ManifestRow | None,
) -> tuple[list[dict[str, object]], PartAudit]:
    """Apply historical change cards and return current rows plus a part audit."""
    converted = [
        (source, row)
        for source in source_rows
        if (row := _historical_row(source, metadata=metadata)) is not None
    ]
    if not converted:
        raise ValueError("historical part has no usable rows")
    filename = str(converted[0][1]["source_filename"])
    ac_number, part_number = _canonical_key(filename)
    base = [
        row
        for source, row in converted
        if source.get("change", "").strip().casefold() in {"", "added"}
    ]
    current, duplicate_source_records, conflicting_source_duplicates = (
        _collapse_source_duplicates(base)
    )
    rows = [row for _, row in converted]
    corrections = [row for row in rows if row["corrected"]]
    deletions = [row for row in rows if row["deleted"]]
    unmatched_corrections = 0
    for correction in corrections:
        target = _event_target(correction, current)
        if target is None:
            unmatched_corrections += 1
            target = correction.copy()
            target["corrected"] = True
            target["deleted"] = False
            current.append(target)
        else:
            for field in (
                "elector_name",
                "father_or_husband_name",
                "relationship",
                "house_no",
                "age",
                "sex",
            ):
                if correction.get(field) not in {None, ""}:
                    target[field] = correction[field]
            target["corrected"] = True
    unmatched_deletions = 0
    for deletion in deletions:
        target = _event_target(deletion, current)
        if target is None:
            unmatched_deletions += 1
            target = deletion.copy()
            current.append(target)
        target["deleted"] = True

    active = [row for row in current if not row["deleted"]]
    totals = {row["net_electors_total"] for row in rows}
    males = {row["net_electors_male"] for row in rows}
    females = {row["net_electors_female"] for row in rows}
    thirds = {row["net_electors_third_gender"] for row in rows}

    def one(values: set[object]) -> int | None:
        present = {int(value) for value in values if isinstance(value, int)}
        return present.pop() if len(present) == 1 else None

    printed_total = one(totals)
    printed_male = one(males)
    printed_female = one(females)
    printed_third = one(thirds)
    active_male = sum(row["sex"] == "Male" for row in active)
    active_female = sum(row["sex"] == "Female" for row in active)
    active_third = sum(row["sex"] == "Third Gender" for row in active)
    serials = [int(row["number"]) for row in current]
    duplicate_serials = len(serials) - len(set(serials))
    active_match = printed_total is not None and len(active) == printed_total
    sex_match = bool(
        printed_male is not None
        and printed_female is not None
        and printed_third is not None
        and (active_male, active_female, active_third)
        == (printed_male, printed_female, printed_third)
    )
    status = (
        "matched"
        if active_match and sex_match
        else ("sex-mismatch" if active_match else "control-mismatch")
    )
    return current, PartAudit(
        source_filename=filename,
        filename=str(converted[0][1]["filename"]),
        assembly_constituency=ac_number,
        part_number=part_number,
        district=str(converted[0][1]["district"]),
        pdf_bytes=0,
        pdf_sha256="",
        pages=0,
        start_serial=min(serials, default=None),
        end_serial=max(serials, default=None),
        printed_male=printed_male,
        printed_female=printed_female,
        printed_third_gender=printed_third,
        printed_total=printed_total,
        mother_records=sum(row["roll_section"] == "main" for row in current),
        addition_records=sum(row["roll_section"] == "addition" for row in current),
        deletion_records=len(deletions),
        correction_records=len(corrections),
        parsed_records=len(current),
        active_records=len(active),
        active_male=active_male,
        active_female=active_female,
        active_third_gender=active_third,
        sex_inferred_records=0,
        records_with_name=sum(bool(row["elector_name"]) for row in current),
        records_with_relative=sum(
            bool(row["father_or_husband_name"]) for row in current
        ),
        records_with_epic=sum(bool(row["id"]) for row in current),
        records_with_house=sum(bool(row["house_no"]) for row in current),
        records_with_valid_age=sum(
            isinstance(row["age"], int) and 18 <= int(row["age"]) <= 120
            for row in current
        ),
        duplicate_source_records=duplicate_source_records,
        conflicting_source_duplicates=conflicting_source_duplicates,
        duplicate_serials=duplicate_serials,
        unmatched_deletions=unmatched_deletions,
        unmatched_corrections=unmatched_corrections,
        boxes_match_serial_range=False,
        active_match_total=active_match,
        sex_match=sex_match,
        serial_contiguous=bool(serials)
        and sorted(serials) == list(range(min(serials), min(serials) + len(serials))),
        control_difference=(len(active) - printed_total if printed_total else None),
        status=status,
        error=None,
    )


def iter_historical_parts(stream: TextIO) -> Iterator[list[dict[str, str]]]:
    """Yield contiguous filename groups from concatenated historical CSVs."""
    reader = csv.reader(stream)
    header: list[str] | None = None
    current_filename = ""
    current: list[dict[str, str]] = []
    seen: set[str] = set()
    for values in reader:
        if values and values[0] == "number":
            header = values
            continue
        if header is None or not values or len(values) != len(header):
            continue
        row = dict(zip(header, values, strict=True))
        filename = row["filename"].strip().upper()
        if current and filename != current_filename:
            if current_filename in seen:
                raise ValueError(f"noncontiguous historical part: {current_filename}")
            seen.add(current_filename)
            yield current
            current = []
        current_filename = filename
        current.append(row)
    if current:
        if current_filename in seen:
            raise ValueError(f"noncontiguous historical part: {current_filename}")
        yield current


def build_recovery(
    historical_stream: TextIO,
    fresh_records_path: Path,
    fresh_parts_path: Path,
    manifest_path: Path,
    records_path: Path,
    parts_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    """Write one hybrid recovery, preferring every available source-PDF reparse."""
    manifest, unavailable = load_manifest(manifest_path)
    by_filename = {
        f"{metadata.filename}.PDF".upper(): metadata for metadata in manifest.values()
    }
    fresh_parts = pq.read_table(fresh_parts_path)
    fresh_names = {
        f"{name}.PDF".upper() for name in fresh_parts.column("filename").to_pylist()
    }
    records_path.parent.mkdir(parents=True, exist_ok=True)
    parts_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_records = records_path.with_suffix(records_path.suffix + ".tmp")
    temporary_parts = parts_path.with_suffix(parts_path.suffix + ".tmp")
    status_counts: Counter[str] = Counter()
    method_files: Counter[str] = Counter()
    method_active: Counter[str] = Counter()
    method_printed: Counter[str] = Counter()
    seen_names: set[str] = set()
    with (
        pq.ParquetWriter(
            temporary_records, FINAL_RECORD_SCHEMA, compression="zstd"
        ) as record_writer,
        pq.ParquetWriter(
            temporary_parts, FINAL_PART_SCHEMA, compression="zstd"
        ) as part_writer,
    ):
        fresh_file = pq.ParquetFile(fresh_records_path)
        for batch in fresh_file.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            age = table.column("age")
            valid_age = pc.and_(pc.greater_equal(age, 18), pc.less_equal(age, 120))
            table = table.set_column(
                table.schema.get_field_index("age"),
                "age",
                pc.if_else(valid_age, age, pa.nulls(table.num_rows, type=pa.int16())),
            )
            table = table.append_column(
                "record_source",
                pa.array(["source-pdf"] * table.num_rows, type=pa.string()),
            )
            record_writer.write_table(table.cast(FINAL_RECORD_SCHEMA))
        fresh_part_rows = []
        for row in fresh_parts.to_pylist():
            row["record_source"] = "source-pdf"
            fresh_part_rows.append(row)
            seen_names.add(f"{row['filename']}.PDF".upper())
            method_files["source-pdf"] += 1
            method_active["source-pdf"] += row["active_records"]
            method_printed["source-pdf"] += row["printed_total"] or 0
            status_counts[row["status"]] += 1
        _write_table(part_writer, fresh_part_rows, FINAL_PART_SCHEMA)

        record_buffer: list[dict[str, object]] = []
        part_buffer: list[dict[str, object]] = []
        for source_rows in iter_historical_parts(historical_stream):
            filename = source_rows[0]["filename"].strip().upper()
            if filename in fresh_names:
                continue
            rows, audit = resolve_historical_part(
                source_rows, metadata=by_filename.get(filename)
            )
            for row in rows:
                row["record_source"] = "historical-parse"
            record_buffer.extend(rows)
            audit_row = asdict(audit)
            audit_row["record_source"] = "historical-parse"
            part_buffer.append(audit_row)
            seen_names.add(filename)
            method_files["historical-parse"] += 1
            method_active["historical-parse"] += audit.active_records
            method_printed["historical-parse"] += audit.printed_total or 0
            status_counts[audit.status] += 1
            if len(record_buffer) >= 100_000:
                _write_table(record_writer, record_buffer, FINAL_RECORD_SCHEMA)
                record_buffer.clear()
            if len(part_buffer) >= 256:
                _write_table(part_writer, part_buffer, FINAL_PART_SCHEMA)
                part_buffer.clear()
        _write_table(record_writer, record_buffer, FINAL_RECORD_SCHEMA)
        _write_table(part_writer, part_buffer, FINAL_PART_SCHEMA)
    temporary_records.replace(records_path)
    temporary_parts.replace(parts_path)
    manifest_names = set(by_filename)
    summary: dict[str, object] = {
        "files": sum(method_files.values()),
        "active_records": sum(method_active.values()),
        "printed_records": sum(method_printed.values()),
        "difference": sum(method_active.values()) - sum(method_printed.values()),
        "files_by_record_source": dict(sorted(method_files.items())),
        "active_records_by_record_source": dict(sorted(method_active.items())),
        "printed_records_by_record_source": dict(sorted(method_printed.items())),
        "statuses": dict(sorted(status_counts.items())),
        "manifest": {
            "available_english_pdfs": len(manifest),
            "unavailable_english_pdfs": unavailable,
            "missing_from_both_sources": len(manifest_names - seen_names),
            "missing_from_both_source_examples": sorted(manifest_names - seen_names)[
                :25
            ],
            "not_in_manifest": len(seen_names - manifest_names),
            "not_in_manifest_examples": sorted(seen_names - manifest_names)[:25],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    """Build the Andhra recovery from historical CSV on standard input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-records", type=Path, required=True)
    parser.add_argument("--fresh-parts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = build_recovery(
        sys.stdin,
        args.fresh_records,
        args.fresh_parts,
        args.manifest,
        args.records,
        args.parts,
        args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
