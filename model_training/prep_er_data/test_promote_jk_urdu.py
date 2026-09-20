"""Test calibrated Urdu promotion on a conserved source spine."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from model_training.prep_er_data.promote_jk_urdu import promote


def write(path, rows, schema=None):
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_promote_fills_only_missing_fields_and_preserves_source_rows(tmp_path):
    base = tmp_path / "base.parquet"
    fields = {
        "filename": "a.pdf",
        "source_sha256": "a" * 64,
        "event_key": "a:1",
        "event_type": "base",
        "supplement": None,
        "number": "1",
        "number_raw": "1",
        "id": "ID",
        "elector_name": None,
        "name_candidate": None,
        "name_issue": "missing",
        "relative_name": None,
        "relative_candidate": None,
        "relative_issue": "missing",
        "relative_type": None,
        "house_no": None,
        "age": None,
        "sex_candidate": None,
        "header_raw": None,
        "raw_cell": None,
        "candidate_cell": None,
        "deletion_stamp_raw": None,
        "entry_event_key": "a:1",
        "page": 3,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "assembly_eligible": True,
        "deleted_stamp": False,
        "active": True,
        "change_event_keys": [],
    }
    known = {
        **fields,
        "event_key": "a:2",
        "entry_event_key": "a:2",
        "number": "2",
        "number_raw": "2",
        "supplement": "base",
        "elector_name": "known",
        "name_candidate": "known",
        "name_issue": None,
        "relative_name": "relative",
        "relative_candidate": "relative",
        "relative_issue": None,
        "relative_type": "باپ",
        "house_no": "1",
        "age": "30",
        "sex_candidate": "M",
        "header_raw": "header",
        "raw_cell": "raw",
        "candidate_cell": "candidate",
        "deletion_stamp_raw": "none",
    }
    write(base, [fields, known])
    overlay = tmp_path / "overlay.parquet"
    write(
        overlay,
        [{**fields, "elector_name": "الف", "name_candidate": "الف"}, known],
    )
    candidates = tmp_path / "candidates.parquet"
    write(
        candidates,
        [
            {
                "filename": "a.pdf",
                "entry_event_key": "a:1",
                "role": "relative",
                "page": 3,
                "region": [0.0, 0.0, 1.0, 1.0],
                "candidate": "ب",
                "direct_ocr_read": False,
                "direct_ocr_candidate": None,
                "review_status": "calibrated_vote_candidate",
            }
        ],
    )
    relationships = tmp_path / "relationships.parquet"
    write(
        relationships,
        [
            {
                "filename": "a.pdf",
                "entry_event_key": "a:1",
                "source_sha256": "a" * 64,
                "page": 3,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "relationship": "باپ",
            }
        ],
    )
    word_audit = tmp_path / "word_audit.json"
    word_audit.write_text(json.dumps({"control_test": {"exact_fraction": 0.95}}))
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text(
        json.dumps(
            {
                "summary": {"inventory_rows": 2, "accepted_urdu_names": 0},
                "parts": [
                    {
                        "filename": "a.pdf",
                        "source_sha256": "a" * 64,
                        "counts": {
                            "inventory": 2,
                            "active_total": 2,
                            "active_npr": 0,
                            "active_assembly": 2,
                        },
                    },
                    {
                        "filename": "empty.pdf",
                        "source_sha256": "b" * 64,
                        "counts": {
                            "inventory": None,
                            "active_total": None,
                            "active_npr": None,
                            "active_assembly": None,
                        },
                    },
                ],
                "script_sha256": {"jk_urdu_rows.py": "1" * 64},
            }
        )
    )
    result = promote(
        base,
        [overlay],
        candidates,
        relationships,
        word_audit,
        source_audit,
        tmp_path / "out",
    )
    row = next(
        row
        for row in pq.read_table(tmp_path / "out/inventory.parquet").to_pylist()
        if row["entry_event_key"] == "a:1"
    )
    assert row["elector_name"] == "الف"
    assert row["relative_name"] == "ب"
    assert row["relative_type"] == "باپ"
    assert row["name_issue"] is None
    assert row["relative_issue"] is None
    assert result["checks"]["immutable_rows_changed"] == 0
    assert result["parts"][0]["active_assembly"] == 2
    assert result["checks"]["source_parts_without_inventory"] == 1
