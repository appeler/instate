"""Build auditable deposits from reconciled English searchable-roll exports.

Run with the sibling upnaam source on PYTHONPATH. This program never downloads
records or modifies the national training inputs. Input CSVs must already have
main/supplement events reconciled by the state parser.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import tarfile
import textwrap
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

POSITION_POLICIES = {"andaman": "last", "dadra": "first"}


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def canonical_input(value):
    return re.sub("[^a-z]", "", (value or "").lower())


def prepare_records(rows, revision):
    """Preserve serial collisions and raw strings; never make EPIC a primary key."""
    seen = Counter()
    output = []
    for raw in rows:
        row = dict(raw)
        filename = Path(row["filename"]).name
        key = (filename, row["number"])
        seen[key] += 1
        row.update(
            record_id=f"{revision}:{filename}:{row['number']}:{seen[key]}",
            source_revision=revision,
            source_filename=filename,
            source_occurrence=seen[key],
            deleted=row["change"] == "deleted",
            relationship=row.get("relative_type", ""),
        )
        output.append(row)
    return output


def count_recorded(bundle):
    """Count only the selected recorded candidate, not relative-only inference."""
    candidates = {r["candidate_id"]: r for r in bundle.candidates.to_pylist()}
    output, counts = [], Counter()
    for resolution in bundle.resolutions.to_pylist():
        selected = candidates.get(resolution["selected_candidate_id"], {})
        recorded = candidates.get(resolution["recorded_candidate_id"], {})
        latin = recorded.get("surname_latin_normalized")
        model_input = canonical_input(latin)
        eligible = (
            resolution["resolution_status"] == "recorded_selected"
            and resolution["recorded_candidate_id"]
            == resolution["selected_candidate_id"]
            and bool(model_input)
        )
        if eligible:
            counts[model_input] += 1
        output.append(
            {
                **resolution,
                "surname_raw": selected.get("surname_raw"),
                "surname_latin_normalized": selected.get("surname_latin_normalized"),
                "surname_canonical": selected.get("surname_canonical"),
                "recorded_surname_latin_normalized": latin,
                "model_input": model_input if eligible else None,
                "training_eligible": eligible,
            }
        )
    return output, counts


def pdf_index(path):
    import pymupdf

    ids, texts = defaultdict(set), []
    with pymupdf.open(path) as document:
        for page in document:
            text = page.get_text()
            texts.append(text)
            for epic in set(re.findall(r"\b[A-Z]{1,5}[0-9]{5,}\b", text)):
                ids[epic].add(page.number + 1)
    return ids, texts


def crop_audit(rows, pdfs, output, prefix):
    """Sample all parsed rows, retaining unlocated and ambiguous sample results."""
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont

    population = sorted(rows, key=lambda row: row["record_id"])
    samples = random.Random(7).sample(population, min(30, len(population)))
    font = ImageFont.load_default(size=17)
    panels, audit = [], []
    for row in samples:
        panel = Image.new("RGB", (1600, 300), "white")
        draw = ImageDraw.Draw(panel)
        locations = []
        with pymupdf.open(pdfs[row["source_filename"]]) as document:
            query = row["id"].strip() or row["elector_name"].strip()
            if query:
                for page in document:
                    for rect in page.search_for(query):
                        locations.append((page.number, rect))
            if len(locations) == 1:
                page_no, rect = locations[0]
                page = document[page_no]
                clip = pymupdf.Rect(
                    0,
                    max(0, rect.y0 - 45),
                    page.rect.width,
                    min(page.rect.height, rect.y1 + 100),
                )
                boxes = [
                    item[1]
                    for drawing in page.get_drawings()
                    for item in drawing["items"]
                    if item[0] == "re"
                    and item[1].contains(rect)
                    and item[1].height > 35
                    and item[1].width > 60
                ]
                if boxes:
                    clip = min(boxes, key=lambda box: box.width * box.height)
                pix = page.get_pixmap(
                    clip=clip, matrix=pymupdf.Matrix(4, 4), alpha=False
                )
                crop = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                crop.thumbnail((1000, 290))
                panel.paste(crop, (0, 0))
                status = "located"
                bbox = list(clip)
            else:
                status = "ambiguous" if locations else "not_located"
                bbox = None
                draw.text(
                    (20, 80),
                    f"Source crop {status}; inspect PDF manually",
                    font=font,
                    fill="black",
                )
        labels = [
            row["source_filename"],
            f"serial={row['number']} status={row['change'] or 'main'}",
            f"name: {row['elector_name']}",
            f"relative: {row['father_or_husband_name']}",
            f"relation={row['relationship']} house={row['house_no']}",
            f"age={row['age']} sex={row['sex']}",
        ]
        y = 8
        for label in labels:
            for line in textwrap.wrap(label, width=53):
                draw.text((1010, y), line, font=font, fill="black")
                y += 23
        audit.append(
            {
                "record_id": row["record_id"],
                "status": status,
                "matches": len(locations),
                "page": locations[0][0] + 1 if len(locations) == 1 else None,
                "bbox": bbox,
                "review_status": "pending_visual_review",
            }
        )
        panels.append(panel)
    combined = Image.new("RGB", (1600, len(panels) * 300), "white")
    for i, panel in enumerate(panels):
        combined.paste(panel, (0, i * 300))
    combined.save(output / f"{prefix}_crop_audit.png")
    for start in range(0, len(panels), 5):
        combined.crop((0, start * 300, 1600, min(start + 5, len(panels)) * 300)).save(
            output / f"{prefix}_crop_audit_{start // 5 + 1}.png"
        )
    write_json(
        output / f"{prefix}_crop_audit.json",
        {"seed": 7, "population": "all parsed rows", "samples": audit},
    )


def build_deposit(csv_path, pdf_dir, output, state, year, revision, target_count):
    from upnaam import LADDER_REVISION, LadderPolicy, NameRecord, resolve_name_records

    from instate.coverage import adjust_surname_counts

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with csv_path.open(newline="") as stream:
        rows = prepare_records(list(csv.DictReader(stream)), revision)
    if not rows:
        raise ValueError("The parser export is empty")
    pdfs = {}
    for path in sorted(pdf_dir.rglob("*.pdf")):
        if path.name in pdfs:
            raise ValueError(f"Ambiguous PDF basename: {path.name}")
        pdfs[path.name] = path
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_filename"]].append(row)
    if set(groups) != set(pdfs):
        raise ValueError(
            "Parsed/source PDF inventories differ; investigate missing or extra parts"
        )
    output.mkdir(parents=True)
    (output / ".gitignore").write_text("*\n!.gitignore\n")
    prefix = f"{state}_{year}"
    position = POSITION_POLICIES[state]
    policy = LadderPolicy(
        f"{state}-english-recorded-v2"
        if state == "dadra"
        else f"{state}-english-recorded-v1",
        position=position,
        relative_position=position,
    )
    counts, statuses, reasons = Counter(), Counter(), Counter()
    parts, source_manifest = [], []
    writers = {}
    with ExitStack() as stack:
        for index, (filename, group) in enumerate(sorted(groups.items()), 1):
            pdf = pdfs[filename]
            epic_pages, pages = pdf_index(pdf)
            source_manifest.append(
                {
                    "filename": filename,
                    "bytes": pdf.stat().st_size,
                    "sha256": digest(pdf),
                    "pages": len(pages),
                }
            )
            for row in group:
                locations = sorted(epic_pages.get(row["id"].strip(), set()))
                row["source_pages"] = locations
                row["source_page"] = locations[0] if len(locations) == 1 else None
                row["source_location_status"] = (
                    "unique_epic"
                    if len(locations) == 1
                    else "ambiguous_epic"
                    if locations
                    else "unlocated"
                )
            active = [r for r in group if not r["deleted"]]
            records = [
                NameRecord(
                    r["record_id"],
                    revision,
                    state,
                    str(year),
                    r["elector_name"],
                    relative_name=r["father_or_husband_name"],
                    relationship=r["relationship"],
                )
                for r in active
            ]
            bundle = resolve_name_records(records, policy=policy)
            surname_rows, part_counts = count_recorded(bundle)
            counts.update(part_counts)
            statuses.update(r["resolution_status"] for r in surname_rows)
            reasons.update(code for r in surname_rows for code in r["reason_codes"])
            control_values = {
                r["net_electors_total"].replace(",", "").strip() for r in group
            }
            if len(control_values) != 1 or not next(iter(control_values)).isdigit():
                raise ValueError(f"Invalid or inconsistent net total: {filename}")
            printed = int(next(iter(control_values)))
            parts.append(
                {
                    "filename": filename,
                    "part_no": group[0]["part_no"],
                    "parsed": len(group),
                    "active": len(active),
                    "deleted": len(group) - len(active),
                    "printed_net": printed,
                    "active_minus_printed": len(active) - printed,
                    "recorded_surname_count": sum(part_counts.values()),
                }
            )
            tables = {
                "": pa.Table.from_pylist(group),
                "_surnames": pa.Table.from_pylist(surname_rows),
                "_surname_resolutions": bundle.resolutions,
                "_surname_candidates": bundle.candidates,
                "_surname_evidence": bundle.evidence,
            }
            for suffix, table in tables.items():
                # Explicit null columns can acquire a type in later parts.
                if suffix == "":
                    schema = pa.schema(
                        [
                            pa.field(
                                k,
                                pa.bool_()
                                if k == "deleted"
                                else pa.int64()
                                if k in {"source_occurrence", "source_page"}
                                else pa.list_(pa.int64())
                                if k == "source_pages"
                                else pa.string(),
                            )
                            for k in group[0]
                        ]
                    )
                    table = pa.Table.from_pylist(group, schema=schema)
                elif suffix == "_surnames":
                    extra = [
                        pa.field(
                            k, pa.bool_() if k == "training_eligible" else pa.string()
                        )
                        for k in surname_rows[0]
                        if k not in bundle.resolutions.column_names
                    ]
                    table = pa.Table.from_pylist(
                        surname_rows,
                        schema=pa.schema([*bundle.resolutions.schema, *extra]),
                    )
                if suffix not in writers:
                    writers[suffix] = stack.enter_context(
                        pq.ParquetWriter(
                            output / f"{prefix}{suffix}.parquet",
                            table.schema,
                            compression="zstd",
                        )
                    )
                writers[suffix].write_table(table)
            if index % 25 == 0:
                print(
                    json.dumps(
                        {
                            "state": state,
                            "parts_done": index,
                            "parts_total": len(groups),
                        }
                    ),
                    flush=True,
                )
    pq.write_table(
        pa.Table.from_pylist(parts),
        output / f"{prefix}_parts.parquet",
        compression="zstd",
    )
    write_json(output / f"{prefix}_pdfs_manifest.json", source_manifest)
    with tarfile.open(output / f"{prefix}_pdfs.tar.gz", "w:gz") as archive:
        for filename, path in sorted(pdfs.items()):
            archive.add(path, arcname=f"{prefix}_pdfs/{filename}", recursive=False)
    active = [r for r in rows if not r["deleted"]]
    pd.DataFrame(active).groupby(
        ["elector_name", "father_or_husband_name"], dropna=False
    ).size().rename("n_times").reset_index().rename(
        columns={
            "elector_name": "english_name",
            "father_or_husband_name": "father_husband_name",
        }
    ).to_csv(
        output / f"names_{state}.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    observed = pd.DataFrame(
        [
            {"state": state, "year": str(year), "surname": name, "observed_count": n}
            for name, n in sorted(counts.items())
        ]
    )
    observed.to_parquet(output / f"{prefix}_surname_counts.parquet", index=False)
    adjustment = adjust_surname_counts(
        observed,
        pd.DataFrame(
            [{"state": state, "year": str(year), "target_count": target_count}]
        ),
        strata=["state", "year"],
        source_revision=revision,
        target_revision=f"{revision}:printed-net",
        assumption="mcar",
    )
    adjustment.estimates.to_parquet(
        output / f"{prefix}_surname_counts_mcar.parquet", index=False
    )
    adjustment.diagnostics.to_parquet(
        output / f"{prefix}_mcar_diagnostics.parquet", index=False
    )
    parse_audit = {
        "source_revision": revision,
        "input_csv_sha256": digest(csv_path),
        "parts": len(parts),
        "parsed": len(rows),
        "active": len(active),
        "printed_net": sum(p["printed_net"] for p in parts),
        "mcar_target": target_count,
        "exact_parts": sum(p["active_minus_printed"] == 0 for p in parts),
        "absolute_part_error": sum(abs(p["active_minus_printed"]) for p in parts),
        "residual_parts": [p for p in parts if p["active_minus_printed"]],
        "source_location_status": dict(
            Counter(r["source_location_status"] for r in rows)
        ),
    }
    if parse_audit["printed_net"] != target_count:
        raise ValueError(
            "MCAR target does not equal edition-matched printed part totals"
        )
    write_json(output / f"{prefix}_parse_audit.json", parse_audit)
    write_json(
        output / f"{prefix}_surnames_audit.json",
        {
            "source_revision": revision,
            "ladder_revision": LADDER_REVISION,
            "policy_revision": policy.revision,
            "active": len(active),
            "recorded_eligible": sum(counts.values()),
            "statuses": dict(statuses),
            "reason_codes": dict(reasons),
            "p_correct": None,
            "household_policy": "disabled: searchable export lacks reliable section identifiers",
            "position_policy": f"{position} content token; upnaam evidence and conflict rules take precedence",
            "training_policy": "recorded_selected only; relative-only candidates excluded",
            "mcar_policy": "separate estimates, state/edition MCAR, before cell suppression; no default-model weighting",
        },
    )
    crop_audit(rows, pdfs, output, prefix)
    dictionary = {
        suffix or "parsed": str(pq.read_schema(output / f"{prefix}{suffix}.parquet"))
        for suffix in writers
    }
    write_json(output / "SCHEMA.json", dictionary)
    (output / f"{prefix}_README.md").write_text(
        f"# {state.title()} electoral roll {year}\n\n"
        f"Source revision: `{revision}`. {len(pdfs)} PDFs; {len(rows):,} parsed rows, "
        f"including {len(rows) - len(active):,} deleted rows; {len(active):,} active records.\n\n"
        "## Files and handling\n\n"
        "The parsed Parquet preserves every CSV field as text, adding stable record IDs, deletion flags, "
        "relationship types, source filenames, and 1-based candidate pages. A single source page is set "
        "only for a unique EPIC match. Original CSV values are not normalized in place. "
        "Serial numbers and EPICs are not unique primary keys. Corrections reflect the parser's reconciled export, "
        "not a complete historical event ledger; source PDFs preserve the original entries.\n\n"
        "The parts table and parse audit retain printed totals and discrepancies without forcing agreement. "
        "The PDF archive and manifest provide the complete source set. Crop audits sample 30 parsed rows "
        "with seed 7; an unlocated crop remains in the sample. Fill rates and reconciliation are not accuracy.\n\n"
        "The surname table contains one row per active record, including abstentions. Separate resolution, "
        "candidate, and evidence tables retain upnaam's ladder, raw spans, reason codes, and matching traces. "
        f"No surname confidence is claimed. {position.title()}-token position is an explicit fallback policy, not a "
        "universal naming convention. Household grouping is disabled because sections are unavailable. "
        "Relative-only inference is retained for analysis but excluded from observed training counts.\n\n"
        "Observed surname counts are unsuppressed research inputs. MCAR counts and diagnostics are separate, "
        "using printed net totals for this edition. MCAR is an assumption, not a finding; it cannot invent "
        "unseen surnames or repair wrong selections. National training uses observed counts and its existing "
        "minimum observed support, never these weighted estimates.\n\n"
        "## Data dictionary\n\n"
        "`SCHEMA.json` records typed table schemas. `record_id` includes source revision, filename, original "
        "serial and occurrence; `deleted` means parser change=deleted; `source_pages` retains all EPIC page "
        "matches; `recorded_candidate_id` and `selected_candidate_id` link to candidates; `training_eligible` "
        "requires a selected recorded Latin surname; `model_input` is lowercase ASCII letters only. "
        "Raw blank strings remain blank; unresolved derived values are null.\n\n"
        "## Access and use\n\n"
        "These artifacts contain personal administrative records and sparse name counts. Keep them private "
        "or access-controlled; do not place row-level examples in public package documentation. Aggregate "
        "research only, not individual identity inference or consequential decisions. Neither electorate "
        "coverage nor surname selection constitutes a population census.\n"
    )
    write_json(
        output / f"{prefix}_build_manifest.json",
        {
            "source_revision": revision,
            "builder_sha256": digest(__file__),
            "input_csv_sha256": digest(csv_path),
            "state": state,
            "year": str(year),
            "target_count": target_count,
            "policy_revision": policy.revision,
            "ladder_revision": LADDER_REVISION,
        },
    )
    for algorithm, name in [("md5", "MD5SUMS"), ("sha256", "SHA256SUMS")]:
        files = sorted(
            p
            for p in output.iterdir()
            if p.is_file() and p.name not in {"MD5SUMS", "SHA256SUMS", ".gitignore"}
        )
        (output / name).write_text(
            "".join(f"{digest(p, algorithm)}  {p.name}\n" for p in files)
        )
    print(json.dumps(parse_audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=["andaman", "dadra"], required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-count", type=int, required=True)
    args = parser.parse_args()
    build_deposit(
        args.csv,
        args.pdf_dir,
        args.output,
        args.state,
        args.year,
        args.source_revision,
        args.target_count,
    )


if __name__ == "__main__":
    main()
