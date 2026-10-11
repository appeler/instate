"""Guard the full-roll schema, demographic joins and explicit missingness."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_training.prep_er_data.build_jk_rolls import (
    BASE_COLUMNS,
    IDENTITY,
    SCHEMA,
    SEX,
    build,
    closing_controls,
    consolidate_demographics,
    convert_row,
    digest,
    inventory_batches,
    isolated_connection,
    station_metadata,
    verified_closing_parts,
    verified_edition_links,
    verified_overlay,
)
from model_training.prep_er_data.jk_hindi_rows import sex_label


@pytest.mark.parametrize("tail", ["आयु 30 लिगं पस्र्ष", "आय 20 लिंग पुस्र्ष"])
def test_hindi_label_variants_are_extracted_without_normalizing_raw(tail):
    assert SEX[sex_label(tail)] == "Male"
    assert sex_label("नाम: महिला") is None
    assert SEX.get(sex_label("आय 20 लिगं �स्र्ष")) is None


@pytest.mark.parametrize("field", IDENTITY)
def test_overlay_cannot_cross_identity_or_source_version(field):
    row = dict.fromkeys(IDENTITY, "same")
    with pytest.raises(ValueError, match="identity conflict"):
        verified_overlay(row, {**row, field: "different"})


def test_conflicting_demographics_fail():
    with pytest.raises(ValueError, match="field conflict"):
        verified_overlay(
            {"entry_event_key": "one", "age": "18"},
            {"entry_event_key": "one", "age": "19"},
        )


def test_source_missingness_and_age_anomalies_survive():
    row = dict.fromkeys(
        [
            *IDENTITY,
            "supplement",
            "name_issue",
            "relative_issue",
            "bbox",
            "elector_name",
            "relative_name",
            "relative_type",
            "house_no",
            "age",
            "sex_candidate",
        ]
    )
    row.update(
        filename="UACA001PS0001.pdf",
        entry_event_key="one",
        event_key="one",
        source_sha256="0" * 64,
        event_type="base",
        deleted_stamp=False,
        change_event_keys=[],
        age="126",
        sex_candidate="مرد",
        active=True,
        assembly_eligible=True,
    )
    station = {
        "ac_number": "1",
        "ps_number": "0001",
        "ac_name": "A",
        "district_name": "D",
        "ps_name": "",
    }
    out = convert_row(
        row,
        "urdu",
        station,
        (None, None, 5),
        None,
        lambda text: (None, "missing_native"),
    )
    assert list(out) == SCHEMA.names
    assert SCHEMA.names[: len(BASE_COLUMNS)] == BASE_COLUMNS
    assert out["age"] == 126
    assert out["age_issue"] == "outside_18_120"
    assert out["sex"] == "Male"
    assert out["husband"] is None
    assert out["polling_station_address"] is None
    assert out["polling_station_name"] is None
    assert out["net_electors_third_gender"] is None
    assert SEX["Third Gen"] == "Third Gender"


def test_bad_closing_controls_do_not_supply_sex_totals():
    assert closing_controls(
        {
            "printed_final_total": 12,
            "closing_summaries": [
                {"numeric_rows": [{"male": 7, "female": 5, "total": 12}]}
            ],
        }
    ) == (None, None, 12)


def test_manifest_filters_literal_null_but_rejects_conflicts(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text("filename,ac_number,ps_number\nnull,1,1\nEACA001PS0001.pdf,1,1\n")
    assert len(station_metadata(path)) == 1
    path.write_text("filename,ac_number,ps_number\nEACA001PS0001.pdf,2,1\n")
    with pytest.raises(ValueError, match="identity conflict"):
        station_metadata(path)


@pytest.mark.parametrize("failure", [None, "identity", "duplicate", "conflict", "null"])
def test_streaming_overlay_join_rejects_corruption(tmp_path, failure):
    base = dict.fromkeys(IDENTITY, "one")
    base.update(house_no=None, age=None, sex_candidate=None)
    dot = {**base, "age": "30"}
    outline = {**base, "house_no": "12", "demographic_issue": None}
    if failure == "identity":
        outline["source_sha256"] = "wrong"
    elif failure == "conflict":
        outline["age"] = "31"
    elif failure == "null":
        outline["entry_event_key"] = None
    pq.write_table(pa.Table.from_pylist([base]), tmp_path / "inventory.parquet")
    pq.write_table(pa.Table.from_pylist([dot]), tmp_path / "demographics.parquet")
    pq.write_table(
        pa.Table.from_pylist([outline] * (2 if failure == "duplicate" else 1)),
        tmp_path / "demographics_outline.parquet",
    )
    if failure:
        with pytest.raises(ValueError, match=r"[Dd]emographic"):
            list(inventory_batches(tmp_path, "urdu"))
    else:
        rows = [r for b in inventory_batches(tmp_path, "urdu") for r in b.to_pylist()]
        assert len(rows) == 1
        assert rows[0]["__demographic_age"] == "30"
        assert rows[0]["__demographic_house_no"] == "12"


@pytest.mark.parametrize("field", ["source_sha256", "printed_final_total"])
def test_closing_audit_cannot_cross_source_identity(tmp_path, field):
    part = {"filename": "one.pdf", "source_sha256": "hash", "printed_final_total": 10}
    path = tmp_path / "closing.json"
    path.write_text(json.dumps({"parts": [part]}))
    assert (
        verified_closing_parts({"one.pdf": part}, path)["one.pdf"][field] == part[field]
    )
    path.write_text(json.dumps({"parts": [{**part, field: "wrong"}]}))
    with pytest.raises(ValueError, match="source identity or total disagreement"):
        verified_closing_parts({"one.pdf": part}, path)


@pytest.mark.parametrize("failure", [None, "hash", "missing_record", "duplicate"])
def test_consolidation_requires_verified_complete_unique_universe(tmp_path, failure):
    source = tmp_path / "source"
    recovery = tmp_path / "recovery"
    source.mkdir()
    recovery.mkdir()
    path = recovery / "UACA001PS0001.pdf.parquet"
    keys = ["one", "one"] if failure == "duplicate" else ["one"]
    pq.write_table(pa.table({"entry_event_key": keys}), path)
    base_keys = ["one", "two"] if failure == "missing_record" else ["one"]
    pq.write_table(
        pa.table({"entry_event_key": base_keys}), source / "inventory.parquet"
    )
    audit = {
        "parts": [
            {
                "filename": "UACA001PS0001.pdf",
                "rows": len(keys),
                "artifact_sha256": "wrong" if failure == "hash" else digest(path),
            }
        ]
    }
    (recovery / "audit.json").write_text(json.dumps(audit))
    if failure:
        with pytest.raises(ValueError, match=r"[Dd]emographic"):
            consolidate_demographics(recovery, source)
    else:
        consolidate_demographics(recovery, source)
        assert pq.read_table(source / "demographics_outline.parquet").num_rows == 1
        assert (source / "demographics_outline_audit.json").is_file()


@pytest.mark.parametrize("remaining", [0, 1])
def test_link_subset_does_not_pass_as_a_complete_union(tmp_path, remaining):
    path = tmp_path / "edition_links.parquet"
    pq.write_table(pa.table({"hindi_entry_event_key": ["one", "two"]}), path)
    audit = {"edition_links": {"sha256": digest(path), "rows": 2, "one_to_one": True}}
    (tmp_path / "edition_links_audit.json").write_text(json.dumps(audit))
    assert verified_edition_links(tmp_path) == 2
    pq.write_table(pa.table({"hindi_entry_event_key": ["one"][:remaining]}), path)
    with pytest.raises(ValueError, match="Edition links disagree"):
        verified_edition_links(tmp_path)


def test_failed_export_never_leaves_a_final_directory(tmp_path, monkeypatch):
    def incomplete(root, output):
        output.mkdir()
        (output / "partial.parquet").write_bytes(b"partial")
        raise ValueError("source disagreement")

    monkeypatch.setattr(
        "model_training.prep_er_data.build_jk_rolls.build_artifacts", incomplete
    )
    output = tmp_path / "final"
    with pytest.raises(ValueError, match="source disagreement"):
        build(tmp_path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".jk-rolls-*"))


@pytest.mark.parametrize("include_outline", [False, True])
def test_streaming_overlay_preserves_disjoint_and_unmatched_keys(
    tmp_path, include_outline
):
    base = dict.fromkeys(IDENTITY, "one")
    base.update(house_no=None, age=None, sex_candidate=None)
    rows = [{**base, "entry_event_key": key} for key in ("one", "two", "three")]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "inventory.parquet")
    pq.write_table(
        pa.Table.from_pylist([{**rows[0], "age": "30"}]),
        tmp_path / "demographics.parquet",
    )
    if include_outline:
        pq.write_table(
            pa.Table.from_pylist(
                [{**rows[1], "house_no": "007", "demographic_issue": "verified_labels"}]
            ),
            tmp_path / "demographics_outline.parquet",
        )
    result = {
        r["entry_event_key"]: r
        for b in inventory_batches(tmp_path, "urdu")
        for r in b.to_pylist()
    }
    assert len(result) == 3
    assert result["one"]["__demographic_age"] == "30"
    assert result["two"]["__demographic_house_no"] == (
        "007" if include_outline else None
    )
    assert result["three"]["__demographic_match"] is False


def test_connections_isolate_spill_files_and_preserve_live_reader():
    with isolated_connection() as first:
        first.execute("SET memory_limit='64MB'")
        first.execute("SET threads=1")
        first_dir = Path(
            first.sql("SELECT current_setting('temp_directory')").fetchone()[0]
        )
        reader = first.sql(
            "SELECT i, repeat(md5(i::VARCHAR), 4) AS payload "
            "FROM range(500000) t(i) ORDER BY i DESC"
        ).to_arrow_reader(batch_size=1000)
        assert any(first_dir.iterdir())
        with isolated_connection() as second:
            second_dir = Path(
                second.sql("SELECT current_setting('temp_directory')").fetchone()[0]
            )
            assert first_dir != second_dir
            second.execute("SET memory_limit='64MB'")
            second.execute("SET threads=1")
            second_reader = second.sql(
                "SELECT i, repeat(md5(i::VARCHAR), 4) AS payload "
                "FROM range(500000) t(i) ORDER BY i DESC"
            ).to_arrow_reader(batch_size=1000)
            assert any(second_dir.iterdir())
            second_reader.close()
        assert not second_dir.exists()
        ids = [value for batch in reader for value in batch.column(0).to_pylist()]
        assert ids == list(range(499999, -1, -1))
    assert not first_dir.exists()
