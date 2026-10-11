"""Export reconciled J&K records in the shared electoral-roll parser schema.

Run with the Instate environment and the local Indicate checkout on PYTHONPATH.
The immutable surname-release inputs remain separate from these enriched rolls.
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from model_training.prep_er_data.jk_hindi_rows import interpret_summary, sex_label


@contextmanager
def isolated_connection():
    """Keep each connection's spill files private until its readers close."""
    with (
        tempfile.TemporaryDirectory(prefix="jk-roll-duckdb-") as temporary,
        duckdb.connect(config={"temp_directory": temporary}) as connection,
    ):
        yield connection


CONTRACT = "jk-2018-electoral-roll-v1"

BASE_COLUMNS = [
    "number",
    "id",
    "elector_name",
    "father_or_husband_name",
    "husband",
    "house_no",
    "age",
    "sex",
    "ac_name",
    "parl_constituency",
    "part_no",
    "year",
    "state",
    "filename",
    "main_town",
    "police_station",
    "mandal",
    "revenue_division",
    "district",
    "pin_code",
    "polling_station_name",
    "polling_station_address",
    "net_electors_male",
    "net_electors_female",
    "net_electors_third_gender",
    "net_electors_total",
    "original_or_amendment",
]
EXTRA_COLUMNS = [
    "record_id",
    "language",
    "assembly_constituency",
    "source_sha256",
    "source_page",
    "source_bbox",
    "entry_event_key",
    "event_key",
    "event_type",
    "supplement",
    "active",
    "assembly_eligible",
    "deleted_stamp",
    "change_event_keys",
    "elector_name_native",
    "elector_name_roman",
    "elector_name_roman_status",
    "relative_name_native",
    "relative_name_roman",
    "relative_name_roman_status",
    "relationship",
    "relationship_label_native",
    "name_issue",
    "relative_issue",
    "age_raw",
    "age_issue",
    "sex_raw",
    "house_source",
    "age_source",
    "sex_source",
    "demographic_issue",
]
INT_COLUMNS = {
    "age",
    "year",
    "assembly_constituency",
    "source_page",
    "net_electors_male",
    "net_electors_female",
    "net_electors_third_gender",
    "net_electors_total",
}
BOOL_COLUMNS = {"husband", "active", "assembly_eligible", "deleted_stamp"}
SCHEMA = pa.schema(
    [
        pa.field(
            name,
            pa.int32()
            if name in INT_COLUMNS
            else pa.bool_()
            if name in BOOL_COLUMNS
            else pa.list_(pa.string())
            if name == "change_event_keys"
            else pa.list_(pa.float64())
            if name == "source_bbox"
            else pa.string(),
        )
        for name in BASE_COLUMNS + EXTRA_COLUMNS
    ]
)
SEX = {
    "Male": "Male",
    "Female": "Female",
    "Third Gend": "Third Gender",
    "Third Gen": "Third Gender",
    "महिला": "Female",
    "पस्र्ष": "Male",
    "पुस्र्ष": "Male",
    "पुरुष": "Male",
    "مرد": "Male",
    "عورت": "Female",
}
RELATION = {
    "Father's Name": "father",
    "Husband's Name": "husband",
    "Mother's Name": "mother",
    "Son in Law": "son_in_law",
    "Self": "self",
    "Adopted Son": "adopted_son",
    "पिता": "father",
    "पति": "husband",
    "माता": "mother",
    "अन्य": "other",
    "باپ": "father",
    "والد": "father",
    "خاوند": "husband",
    "ماں": "mother",
    "والدہ": "mother",
}
IDENTITY = [
    "filename",
    "entry_event_key",
    "source_sha256",
    "event_key",
    "number",
    "id",
    "active",
    "assembly_eligible",
    "page",
]


def digest(path):
    """Hash a file without loading it into memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def consolidate_demographics(recovery, source):
    """Retain a complete, hash-verified outline pass as one private overlay."""
    audit_path = recovery / "audit.json"
    audit = json.loads(audit_path.read_text())
    files = []
    for part in audit["parts"]:
        filename = part["filename"]
        if not re.fullmatch(r"UACA[0-9]+PS[0-9]+\.pdf", filename):
            raise ValueError("Invalid demographic checkpoint filename")
        path = recovery / (filename + ".parquet")
        if digest(path) != part["artifact_sha256"]:
            raise ValueError("Demographic checkpoint hash mismatch")
        if pq.ParquetFile(path).metadata.num_rows != part["rows"]:
            raise ValueError("Demographic checkpoint row count mismatch")
        files.append(str(path))
    if not files or len(files) != len(set(files)):
        raise ValueError("Missing or duplicate demographic checkpoints")
    target = source / "demographics_outline.parquet"
    if target.exists():
        raise FileExistsError(target)
    with isolated_connection() as connection:
        connection.execute("SET memory_limit='1GB'")
        connection.execute("SET threads=2")
        connection.read_parquet(files).create_view("recovered")
        connection.read_parquet(str(source / "inventory.parquet")).create_view("base")
        if (
            connection.sql("""SELECT count(*) FROM (
            SELECT entry_event_key FROM recovered EXCEPT
            SELECT entry_event_key FROM base)""").fetchone()[0]
            or connection.sql(
                """SELECT count(*) FROM (
            SELECT entry_event_key FROM base EXCEPT
            SELECT entry_event_key FROM recovered)"""
            ).fetchone()[0]
        ):
            raise ValueError("Demographic recovery is not the complete inventory")
        if connection.sql("""SELECT count(*)-count(DISTINCT entry_event_key)
            FROM recovered""").fetchone()[0]:
            raise ValueError("Duplicate or missing demographic entry key")
        temporary = target.with_suffix(".parquet.tmp")
        connection.sql(
            "SELECT * FROM recovered ORDER BY entry_event_key"
        ).write_parquet(str(temporary), compression="zstd")
        temporary.replace(target)
    audit["consolidated_sha256"] = digest(target)
    audit["source_audit_sha256"] = digest(audit_path)
    (source / "demographics_outline_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )


def station_metadata(path):
    """Require a single consistent metadata row per genuine source filename."""
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["filename"]
            if not name or name.lower() == "null":
                continue
            match = re.fullmatch(r"[EHU]ACA(\d+)PS(\d+)\.pdf", name)
            if not match or tuple(map(int, match.groups())) != (
                int(row["ac_number"]),
                int(row["ps_number"]),
            ):
                raise ValueError(f"Filename/metadata identity conflict: {name}")
            if name in result and result[name] != row:
                raise ValueError(f"Conflicting station metadata: {name}")
            result[name] = row
    return result


def closing_controls(part):
    """Use only an unambiguous, arithmetically valid closing control."""
    choices = set()
    for summary in part.get("closing_summaries", []):
        rows = summary.get("numeric_rows", [])
        check = interpret_summary(rows)
        if check["arithmetic_valid"] and check["final_total"] == part.get(
            "printed_final_total"
        ):
            final = rows[5]
            choices.add((final["male"], final["female"], final["total"]))
    if len(choices) == 1:
        male, female, total = choices.pop()
        return male, female, total
    return None, None, part.get("printed_final_total")


def verified_closing_parts(parts, audit_path):
    """Join retained closing-page evidence only to identical canonical PDFs."""
    if not audit_path.exists():
        return parts
    closing = json.loads(audit_path.read_text())["parts"]
    by_name = {part["filename"]: part for part in closing}
    if len(by_name) != len(closing):
        raise ValueError("Duplicate closing-control part")
    result = {}
    for name, part in parts.items():
        evidence = by_name.get(name)
        if evidence is None or any(
            part.get(key) != evidence.get(key)
            for key in ("source_sha256", "printed_final_total")
        ):
            raise ValueError("Closing-control source identity or total disagreement")
        result[name] = {
            **part,
            "closing_summaries": evidence.get("closing_summaries", []),
        }
    return result


def verified_edition_links(root):
    """Require the entire accepted link artifact, not merely a one-to-one subset."""
    path = root / "edition_links.parquet"
    evidence = json.loads((root / "edition_links_audit.json").read_text())[
        "edition_links"
    ]
    if (
        evidence.get("one_to_one") is not True
        or evidence.get("sha256") != digest(path)
        or evidence.get("rows") != pq.ParquetFile(path).metadata.num_rows
    ):
        raise ValueError("Edition links disagree with accepted audit")
    return evidence["rows"]


def verified_overlay(base, overlay):
    """Join demographic evidence only to the exact same source card and identity."""
    if any(base.get(key) != overlay.get(key) for key in IDENTITY):
        raise ValueError(
            f"Demographic overlay identity conflict: {base['entry_event_key']}"
        )
    for key in ("house_no", "age", "sex_candidate"):
        if base.get(key) is not None and overlay.get(key) not in (None, base[key]):
            raise ValueError(f"Demographic overlay field conflict: {key}")


def romanizer(root, language):
    """Use local Indicate tables with prior source-review abstentions enforced."""
    from indicate.lookup import Lookup

    if language == "english":
        return lambda text: (text, "source_latin" if text else "missing_native")
    table = Lookup.from_path(root / language / "lookup.tsv.gz")
    if table is None:
        raise ValueError(f"Missing or incompatible {language} Indicate table")
    if language == "hindi":
        reviewed = json.loads((root / language / "romanization_map.json").read_text())
        review = pq.read_table(root / language / "transliteration_review.parquet")
        overrides = {row["native"]: None for row in review.to_pylist()}
        overrides.update(reviewed)
        table = table.with_overrides(overrides)

    @lru_cache(maxsize=262144)
    def convert(text):
        if not text:
            return None, "missing_native"
        latin = table.get_text(text)
        return latin, "lookup_complete" if latin is not None else "unresolved_token"

    return convert


def inventory_batches(source, language):
    """Validate demographic join cardinality and stream it with bounded memory."""
    if language != "urdu":
        yield from pq.ParquetFile(source / "inventory.parquet").iter_batches(
            batch_size=16384
        )
        return
    with isolated_connection() as connection:
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET threads=2")
        connection.read_parquet(str(source / "inventory.parquet")).create_view("base")
        overlays = [source / "demographics.parquet"]
        outline = source / "demographics_outline.parquet"
        if outline.exists():
            overlays.append(outline)
        for index, path in enumerate(overlays):
            connection.read_parquet(str(path)).create_view(f"overlay_{index}")
        tables = ["base", *(f"overlay_{i}" for i in range(len(overlays)))]
        for table in tables:
            bad = connection.sql(
                f"SELECT count(*)-count(DISTINCT entry_event_key), "  # noqa: S608
                "count(*) FILTER(WHERE entry_event_key IS NULL OR entry_event_key='') "
                f"FROM {table}"
            ).fetchone()
            if any(bad):
                raise ValueError("Duplicate or missing demographic join keys")
        comparison = " OR ".join(
            f"d.{key} IS DISTINCT FROM b.{key}" for key in IDENTITY
        )
        for table in tables[1:]:
            bad = connection.sql(
                f"SELECT count(*) FROM {table} d "  # noqa: S608
                "LEFT JOIN base b USING(entry_event_key) "
                f"WHERE {comparison}"
            ).fetchone()[0]
            if bad:
                raise ValueError(
                    "Demographic overlay identity conflict or unmatched key"
                )
        if outline.exists():
            conflict = " OR ".join(
                f"(a.{field} IS NOT NULL AND b.{field} IS NOT NULL "
                f"AND a.{field} IS DISTINCT FROM b.{field})"
                for field in ("house_no", "age", "sex_candidate")
            )
            if connection.sql(
                "SELECT count(*) FROM overlay_0 a "  # noqa: S608
                "JOIN overlay_1 b USING(entry_event_key) "
                f"WHERE {conflict}"
            ).fetchone()[0]:
                raise ValueError("Conflicting demographic evidence")
            connection.sql("""SELECT
                coalesce(a.entry_event_key,b.entry_event_key) AS entry_event_key,
                coalesce(a.house_no,b.house_no) AS house_no,
                coalesce(a.age,b.age) AS age,
                coalesce(a.sex_candidate,b.sex_candidate) AS sex_candidate,
                b.demographic_issue
                FROM overlay_0 a FULL OUTER JOIN overlay_1 b
                USING(entry_event_key)""").create_view("demo")
        else:
            connection.sql("""SELECT entry_event_key,house_no,age,sex_candidate,
                NULL::VARCHAR AS demographic_issue FROM overlay_0""").create_view(
                "demo"
            )
        reader = connection.sql("""SELECT b.*,
            d.entry_event_key IS NOT NULL AS __demographic_match,
            d.house_no AS __demographic_house_no, d.age AS __demographic_age,
            d.sex_candidate AS __demographic_sex_candidate,
            d.demographic_issue AS __demographic_issue
            FROM base b LEFT JOIN demo d USING(entry_event_key)
            ORDER BY b.entry_event_key""").to_arrow_reader(16384)
        yield from reader


def convert_row(row, language, station, controls, overlay, romanize):
    """Map one reconciled inventory row without inferring missing personal fields."""
    if any(
        not isinstance(row[key], str) or not row[key]
        for key in ("entry_event_key", "event_key", "source_sha256", "filename")
    ):
        raise ValueError("Missing source identity")
    if any(not isinstance(row[key], bool) for key in ("active", "assembly_eligible")):
        raise ValueError("Missing activity or eligibility flag")
    if not row["filename"].startswith(
        {"english": "E", "hindi": "H", "urdu": "U"}[language]
    ):
        raise ValueError("Filename language disagrees with edition")
    result = dict.fromkeys(BASE_COLUMNS + EXTRA_COLUMNS)
    for key in (
        "number",
        "id",
        "filename",
        "source_sha256",
        "entry_event_key",
        "event_key",
        "event_type",
        "supplement",
        "active",
        "assembly_eligible",
        "deleted_stamp",
        "change_event_keys",
        "name_issue",
        "relative_issue",
    ):
        result[key] = row[key]
    result.update(
        record_id=f"jk-2018:{language}:{row['entry_event_key']}",
        language=language,
        state="JK",
        year=2018,
        source_page=row["page"],
        source_bbox=row["bbox"],
        assembly_constituency=int(station["ac_number"]),
        part_no=station["ps_number"],
        ac_name=station["ac_name"] or None,
        district=station["district_name"] or None,
        polling_station_name=station["ps_name"] or None,
        net_electors_male=controls[0],
        net_electors_female=controls[1],
        net_electors_total=controls[2],
        original_or_amendment="original"
        if row["event_type"] == "base"
        else "amendment",
        elector_name=row["elector_name"],
        elector_name_native=row["elector_name"],
        father_or_husband_name=row["relative_name"],
        relative_name_native=row["relative_name"],
        relationship_label_native=row["relative_type"],
        demographic_issue=row.get("__demographic_issue"),
    )
    for field, native in (
        ("elector_name", row["elector_name"]),
        ("relative_name", row["relative_name"]),
    ):
        result[field + "_roman"], result[field + "_roman_status"] = romanize(native)
    relation = RELATION.get(row["relative_type"])
    result["relationship"] = relation
    result["husband"] = relation == "husband" if relation else None
    if overlay is not None:
        verified_overlay(row, overlay)
    for field, target in (
        ("house_no", "house"),
        ("age", "age"),
        ("sex_candidate", "sex"),
    ):
        value = row[field]
        source = "accepted_inventory" if value is not None else None
        if value is None and overlay is not None and overlay[field] is not None:
            value, source = overlay[field], "verified_urdu_demographic_overlay"
        if target == "sex" and value is None and language == "hindi":
            value = sex_label(row["candidate_cell"] or "")
            source = "labeled_hindi_card_tail" if value else None
        result[target + "_source"] = source
        if target == "house":
            result["house_no"] = value
        elif target == "sex":
            result["sex_raw"], result["sex"] = value, SEX.get(value)
        else:
            result["age_raw"] = value
            age = int(value) if value and re.fullmatch(r"[0-9]{1,4}", value) else None
            result["age"] = age
            result["age_issue"] = (
                "missing"
                if value is None
                else "unparsed"
                if age is None
                else "outside_18_120"
                if not 18 <= age <= 120
                else None
            )
    return result


def build(root, output):
    """Publish a local export directory only after every build check succeeds."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".jk-rolls-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "rolls"
        build_artifacts(root, staged)
        staged.rename(output)


def build_artifacts(root, output):
    """Build private roll editions, active exports, a union index and audits."""
    from indicate import lookup, normalize

    output.mkdir(parents=True, exist_ok=False)
    stations = station_metadata(root / "polling_stations.csv")
    expected_links = verified_edition_links(root)
    code_paths = {
        "build_jk_rolls.py": Path(__file__),
        "jk_hindi_rows.py": Path(__file__).with_name("jk_hindi_rows.py"),
        "indicate/lookup.py": Path(lookup.__file__),
        "indicate/normalize.py": Path(normalize.__file__),
    }
    audit = {
        "contract": CONTRACT,
        "row_unit": "one reconciled source-edition entry, including inactive and NPR",
        "schema_source": (
            "electoral_rolls_parse_searchable/README.md and scripts/jk_hindi.py"
        ),
        "accuracy": "Not estimated. Coverage and structural checks are not accuracy.",
        "languages": {},
        "inputs": {},
        "script_sha256": {name: digest(path) for name, path in code_paths.items()},
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("duckdb", "pyarrow")},
        },
    }
    for path in sorted(root.rglob("*")):
        if path.is_file():
            audit["inputs"][str(path.relative_to(root))] = digest(path)
    for language in ("english", "hindi", "urdu"):
        source = root / language
        parts = json.loads((source / "source_audit.json").read_text())["parts"]
        source_audit = json.loads((source / "source_audit.json").read_text())
        if source_audit["artifacts"]["inventory.parquet"] != digest(
            source / "inventory.parquet"
        ):
            raise ValueError("Inventory hash does not match its source audit")
        part_map = {p["filename"]: p for p in parts}
        if len(part_map) != len(parts):
            raise ValueError("Duplicate audit part")
        control_parts = verified_closing_parts(
            part_map, source / "closing_controls_audit.json"
        )
        controls = {
            name: closing_controls(part) for name, part in control_parts.items()
        }
        romanize = romanizer(root, language)
        counts, fields, categories, part_counts = (
            Counter(),
            Counter(),
            defaultdict(Counter),
            defaultdict(Counter),
        )
        with pq.ParquetWriter(
            output / f"jk_2018_{language}.parquet", SCHEMA, compression="zstd"
        ) as writer:
            for batch in inventory_batches(source, language):
                rows = []
                for row in batch.to_pylist():
                    part = part_map[row["filename"]]
                    if part["source_sha256"] != row["source_sha256"]:
                        raise ValueError("Source hash disagrees with part audit")
                    overlay = (
                        {
                            **row,
                            **{
                                key: row["__demographic_" + key]
                                for key in ("house_no", "age", "sex_candidate")
                            },
                        }
                        if row.get("__demographic_match")
                        else None
                    )
                    out = convert_row(
                        row,
                        language,
                        stations[row["filename"]],
                        controls[row["filename"]],
                        overlay,
                        romanize,
                    )
                    rows.append(out)
                    counts["rows"] += 1
                    part_counts[row["filename"]]["inventory"] += 1
                    if out["active"]:
                        counts["active"] += 1
                        part_counts[row["filename"]]["active"] += 1
                        part_counts[row["filename"]][out["sex"] or "unknown_sex"] += 1
                        if out["assembly_eligible"]:
                            counts["active_assembly"] += 1
                    for key in SCHEMA.names:
                        fields[key] += out[key] is not None
                    for key in (
                        "sex",
                        "sex_raw",
                        "relationship",
                        "age_issue",
                        "elector_name_roman_status",
                        "relative_name_roman_status",
                    ):
                        categories[key][out[key] or "<null>"] += 1
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        part_report = []
        for name, part in sorted(part_map.items()):
            count = part_counts[name]
            expected = part.get("counts", part)
            if count["inventory"] != expected.get("inventory", 0):
                raise ValueError(
                    f"Inventory row count disagrees with source audit: {name}"
                )
            if count["active"] != expected.get("active_total", 0):
                raise ValueError(
                    f"Active row count disagrees with source audit: {name}"
                )
            male, female, total = controls[name]
            part_report.append(
                {
                    "filename": name,
                    **count,
                    "printed_male": male,
                    "printed_female": female,
                    "printed_total": total,
                    "total_difference": count["active"] - total
                    if total is not None
                    else None,
                    "male_difference": count["Male"] - male
                    if male is not None
                    else None,
                    "female_difference": count["Female"] - female
                    if female is not None
                    else None,
                }
            )
        (output / f"{language}_parts_audit.json").write_text(
            json.dumps(part_report, indent=2) + "\n"
        )
        audit["languages"][language] = {
            **counts,
            "nonnull": dict(fields),
            "categories": dict(categories),
        }
        print(language, dict(counts), flush=True)  # noqa: T201

    with isolated_connection() as con:
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=2")
        for language in ("english", "hindi", "urdu"):
            path = output / f"jk_2018_{language}.parquet"
            con.read_parquet(str(path)).create_view(language)
            rows, keys, nulls = con.execute(
                "SELECT count(*),count(DISTINCT record_id),"  # noqa: S608
                f"count(*) FILTER(WHERE record_id IS NULL) FROM {language}"
            ).fetchone()
            if rows != keys or nulls:
                raise ValueError("Final record identities are not unique")
            quality = con.sql(
                "SELECT count(*) AS rows,"  # noqa: S608
                "count(nullif(id,'')) AS nonempty_elector_ids,"
                "count(DISTINCT nullif(id,'')) AS distinct_elector_ids,"
                "count(*) FILTER (WHERE age_issue='outside_18_120') AS age_anomalies "
                f"FROM {language} WHERE active AND assembly_eligible"
            ).fetchone()
            audit["languages"][language]["active_assembly_quality"] = {
                "rows": quality[0],
                "nonempty_elector_ids": quality[1],
                "distinct_elector_ids": quality[2],
                "repeated_id_excess": quality[1] - quality[2],
                "age_outside_18_120": quality[3],
                "repeated_id_policy": (
                    "Diagnostic only; repeated IDs do not establish duplicate people."
                ),
            }
            coverage = (
                con.sql(
                    "SELECT assembly_constituency,count(*) AS rows,"  # noqa: S608
                    "count(elector_name) AS native_name,"
                    "count(elector_name_roman) AS roman_name,"
                    "count(father_or_husband_name) AS native_relative,"
                    "count(house_no) AS house_no,count(age) AS age,count(sex) AS sex "
                    f"FROM {language} WHERE active AND assembly_eligible "
                    "GROUP BY assembly_constituency ORDER BY assembly_constituency"
                )
                .to_arrow_table()
                .to_pylist()
            )
            audit["languages"][language]["active_assembly_coverage_by_constituency"] = (
                coverage
            )
            target = output / f"jk_2018_{language}_active.parquet"
            con.sql(
                f"SELECT * FROM {language} WHERE active AND assembly_eligible"  # noqa: S608
            ).write_parquet(str(target), compression="zstd")
        con.read_parquet(str(root / "edition_links.parquet")).create_view("links")
        link_counts = con.sql(
            "SELECT count(*), count(DISTINCT urdu_entry_event_key),"
            "count(DISTINCT hindi_entry_event_key) FROM links"
        ).fetchone()
        matched = con.sql("""SELECT count(*) FROM links l
            JOIN hindi h ON h.entry_event_key=l.hindi_entry_event_key
                AND h.filename=l.hindi_filename
            JOIN urdu u ON u.entry_event_key=l.urdu_entry_event_key
                AND u.filename=l.urdu_filename
            WHERE h.active AND h.assembly_eligible
                AND u.active AND u.assembly_eligible""").fetchone()[0]
        if len({*link_counts, matched, expected_links}) != 1:
            raise ValueError("Edition links are not complete and one-to-one")
        con.sql("""SELECT h.record_id, u.record_id AS partner_record_id
            FROM links l JOIN hindi h ON h.entry_event_key=l.hindi_entry_event_key
            JOIN urdu u ON u.entry_event_key=l.urdu_entry_event_key""").write_parquet(
            str(output / "edition_links.parquet"), compression="zstd"
        )
        con.sql("""SELECT record_id,language FROM english
            WHERE active AND assembly_eligible
            UNION ALL SELECT record_id,language FROM hindi
            WHERE active AND assembly_eligible
            UNION ALL SELECT record_id,language FROM urdu u
            WHERE active AND assembly_eligible
            AND NOT EXISTS(SELECT 1 FROM links l
                WHERE l.urdu_entry_event_key=u.entry_event_key)""").write_parquet(
            str(output / "reconciled_active_index.parquet"), compression="zstd"
        )
        con.sql("""SELECT * FROM english WHERE active AND assembly_eligible
            UNION ALL SELECT * FROM hindi WHERE active AND assembly_eligible
            UNION ALL SELECT * FROM urdu u WHERE active AND assembly_eligible
            AND NOT EXISTS(SELECT 1 FROM links l
                WHERE l.urdu_entry_event_key=u.entry_event_key)""").write_parquet(
            str(output / "jk_2018_active.parquet"), compression="zstd"
        )
        audit["edition_reconciliation"] = {
            "links": matched,
            "rows": pq.ParquetFile(
                output / "reconciled_active_index.parquet"
            ).metadata.num_rows,
            "policy": (
                "Keep Hindi for linked entries, all unlinked entries. Partners remain "
                "in edition files; this is an edition union, not verified unique "
                "people statewide."
            ),
        }
        con.read_parquet(str(root / "urdu" / "inventory.parquet")).create_view(
            "base_urdu"
        )
        enriched = output / "urdu_enriched_inventory.parquet"
        con.sql("""SELECT b.* REPLACE(
            r.house_no AS house_no, r.age_raw AS age, r.sex_raw AS sex_candidate)
            FROM base_urdu b JOIN urdu r USING(entry_event_key)
            ORDER BY b.entry_event_key""").write_parquet(
            str(enriched), compression="zstd"
        )
        enriched_audit = json.loads((root / "urdu" / "source_audit.json").read_text())
        enriched_audit["artifacts"] = {"inventory.parquet": digest(enriched)}
        enriched_audit["script_sha256"][Path(__file__).name] = digest(__file__)
        enriched_audit["demographic_enrichment"] = {
            "base_inventory_sha256": digest(root / "urdu" / "inventory.parquet"),
            "overlay_sha256": digest(root / "urdu" / "demographics.parquet"),
            "outline_overlay_sha256": (
                digest(root / "urdu" / "demographics_outline.parquet")
                if (root / "urdu" / "demographics_outline.parquet").exists()
                else None
            ),
            "join_contract": IDENTITY,
            "changed_fields": ["house_no", "age", "sex_candidate"],
            "names_and_row_universe": "unchanged",
        }
        (output / "urdu_enriched_audit.json").write_text(
            json.dumps(enriched_audit, indent=2, ensure_ascii=False) + "\n"
        )
        if audit["script_sha256"] != {
            name: digest(path) for name, path in code_paths.items()
        } or any(
            digest(root / name) != expected
            for name, expected in audit["inputs"].items()
        ):
            raise ValueError("Roll-build inputs or code changed during extraction")
        audit["files"] = {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file()
        }
        (output / "audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
        )
        (output / "SCHEMA.json").write_text(
            json.dumps(
                {
                    "contract": CONTRACT,
                    "base_columns": BASE_COLUMNS,
                    "columns": {field.name: str(field.type) for field in SCHEMA},
                    "null_policy": (
                        "Null denotes unavailable or unsupported evidence, never "
                        "a zero or an inferred field."
                    ),
                    "base_names": (
                        "elector_name and father_or_husband_name retain "
                        "accepted native "
                        "source text; *_roman is a complete lookup rendering or null."
                    ),
                    "printed_controls": (
                        "Closing controls include assembly and NPR entries; "
                        "third-gender "
                        "control is unavailable. Do not sum repeated controls across "
                        "rows."
                    ),
                    "record_identity": (
                        "record_id is edition + stable entry_event_key; "
                        "event_key identifies "
                        "the latest reconciled card. Source pages are 1-based PDF "
                        "pages."
                    ),
                    "unavailable": (
                        "parl_constituency, main_town, police_station, mandal, "
                        "revenue_division, pin_code, polling_station_address and "
                        "third-gender totals remain null; station names are not "
                        "addresses."
                    ),
                    "sex": (
                        "Printed category: Male, Female, Third Gender, or null. "
                        "Unknown spellings remain sex_raw."
                    ),
                    "age": (
                        "Parsed integer retained even outside 18-120; "
                        "age_raw and age_issue preserve anomalies."
                    ),
                    "original_or_amendment": (
                        "original for a latest base card; amendment for a "
                        "latest addition "
                        "or correction; event_type and change_event_keys retain "
                        "details."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        (output / "SHA256SUMS").write_text(
            "".join(
                f"{digest(path)}  {path.name}\n"
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "SHA256SUMS"
            )
        )


def main():
    """Parse paths and build the roll deposit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demographic-recovery", type=Path)
    args = parser.parse_args()
    if args.demographic_recovery:
        consolidate_demographics(args.demographic_recovery, args.root / "urdu")
    build(args.root, args.out)


if __name__ == "__main__":
    main()
