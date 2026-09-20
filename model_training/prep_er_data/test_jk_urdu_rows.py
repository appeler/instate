"""Identity and section-completeness checks for the Urdu source ledger."""

import gzip
import json

import pytest

from model_training.prep_er_data.jk_urdu_rows import (
    isolated_part,
    reconcile_part,
    select_parts,
    source_id,
    structural_payloads,
)


def test_streamed_parts_preserve_order_empty_sources_and_float_coordinates(tmp_path):
    parts = [
        {"filename": "a.pdf", "appearances": 2, "x": 1.25},
        {"filename": "failed.pdf", "appearances": None},
        {"filename": "empty.pdf", "appearances": 0},
        {"filename": "b.pdf", "appearances": 1},
    ]
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"parts": parts}))
    rows = [{"filename": name} for name in ["a.pdf", "a.pdf", "b.pdf"]]
    result = list(structural_payloads(audit, iter(rows)))
    assert [p for _, p in result] == parts
    assert [len(r) for r, _ in result] == [2, 0, 0, 1]
    assert isinstance(result[0][1]["x"], float)


def test_compressed_audit_keeps_the_same_source_evidence(tmp_path):
    path = tmp_path / "audit.json.gz"
    part = {"filename": "one.pdf", "appearances": 1, "x": 2.5}
    with gzip.open(path, "wt") as stream:
        json.dump({"parts": [part]}, stream)
    rows = [{"filename": "one.pdf"}]
    assert list(structural_payloads(path, iter(rows))) == [(rows, part)]


def test_part_selection_preserves_source_order_and_rejects_absent_parts():
    payloads = [([], {"filename": name}) for name in ["a", "b", "c"]]
    assert list(select_parts(iter(payloads), ["c", "a"])) == [payloads[0], payloads[2]]
    with pytest.raises(ValueError, match="Requested parts absent"):
        list(select_parts(iter(payloads), ["missing"]))


@pytest.mark.parametrize(
    ("names", "parts"),
    [
        (["b", "a"], [("a", 1), ("b", 1)]),
        (["a", "b", "a"], [("a", 1), ("b", 1)]),
        (["a"], [("a", 1), ("a", 0)]),
        (["a", "a"], [("a", 1)]),
    ],
)
def test_streamed_parts_reject_mismatched_or_repeated_sources(tmp_path, names, parts):
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"parts": [{"filename": n, "appearances": c} for n, c in parts]})
    )
    with pytest.raises(ValueError, match=r"Structural|Repeated source part"):
        list(structural_payloads(audit, ({"filename": n} for n in names)))


def test_failed_extraction_is_not_retried_or_counted_as_zero(tmp_path):
    source = {**part(), "issues": [{"reason": "TimeoutExpired"}]}
    events, inventory, audit = isolated_part(([], source), tmp_path, tmp_path, 1)
    assert events == inventory == []
    assert audit["parse_status"] == "no_structural_rows"
    assert all(value is None for value in audit["counts"].values())
    assert audit["issues"] == source["issues"]


def test_source_id_uses_only_unique_header_token_and_preserves_original():
    row = {"id": None, "latin_text": "125 lsw1234567\n35 ABC9999999"}
    assert source_id(row) == "LSW1234567"
    assert row["latin_text"].startswith("125 lsw")
    assert source_id({"id": None, "latin_text": "125\nABC9999999"}) is None
    assert source_id({"id": None, "latin_text": "ABC1234567 XYZ7654321"}) is None


def appearance(number, identifier, key, page, *, deleted=False):
    return {
        "filename": "source.pdf",
        "source_sha256": "hash",
        "number": number,
        "id": identifier,
        "latin_text": f"{number} {identifier or ''}",
        "appearance_key": f"source.pdf:{page}:{key}",
        "page": page,
        "deleted_stamp": deleted,
        "stamp_candidate": "DELETED" if deleted else None,
    }


def part():
    return {
        "filename": "source.pdf",
        "source_sha256": "hash",
        "split": "validation",
        "page_sequence": {"valid": True},
        "closing_summaries": [
            {
                "format_supported": True,
                "base_total": 1,
                "addition_total": 1,
                "deletion_total": 1,
                "correction_total": 0,
                "final_total": 1,
            }
        ],
    }


def label(page, kind, component=None):
    marker = (
        []
        if component is None
        else [
            {
                "component": component,
                "label": {"event_type": kind, "assembly_eligible": True},
            }
        ]
    )
    return {
        "page": page,
        "event_type": kind,
        "assembly_eligible": True,
        "markers": marker,
    }


def test_missing_supplement_heading_cannot_pass_by_deduplicating_appearances():
    rows = [
        appearance("1", "ABC1234567", 1, 3),
        appearance("2", "XYZ7654321", 1, 4),
        appearance("1", "ABC1234567", 1, 5),
    ]
    labels = [label(3, "base"), label(4, "base"), label(5, "base")]
    events, inventory, audit = reconcile_part(rows, part(), labels)
    assert len(events) == 3
    assert inventory == []
    assert audit["parse_status"] == "unresolved_sections"
    assert audit["counts"]["active_total"] is None
    assert {i["component"] for i in audit["issues"]} == {1, 2}


def test_wrong_part_closing_withholds_inventory_but_preserves_appearances():
    source = {**part(), "part_identity": {"valid": False}}
    rows = [appearance("1", "ABC1234567", 1, 3)]
    events, inventory, audit = reconcile_part(rows, source, [label(3, "base")])
    assert len(events) == 1
    assert events[0]["id"] == "ABC1234567"
    assert inventory == []
    assert audit["parse_status"] == "unresolved_source"
    assert audit["printed_final_total"] is None
    assert audit["final_total_difference"] is None
    assert audit["issues"] == [{"reason": "printed_part_identity_mismatch"}]


def test_deletion_stamp_and_deletion_section_do_not_remove_two_records():
    rows = [
        appearance("1", "ABC1234567", 1, 3, deleted=True),
        appearance("2", "XYZ7654321", 1, 4),
        appearance("1", "ABC1234567", 1, 5),
    ]
    labels = [label(3, "base"), label(4, "addition", 1), label(5, "deletion", 2)]
    _, inventory, audit = reconcile_part(rows, part(), labels)
    assert len(inventory) == 2
    assert audit["counts"]["active_total"] == 1
    assert audit["final_total_difference"] == 0
    assert not any(audit["event_count_differences"].values())


def test_source_control_disagreement_never_creates_or_drops_records():
    rows = [
        appearance("1", "ABC1234567", 1, 3, deleted=True),
        appearance("2", "XYZ7654321", 1, 4),
        appearance("1", "ABC1234567", 1, 5),
    ]
    labels = [label(3, "base"), label(4, "addition", 1), label(5, "deletion", 2)]
    source = part()
    source["closing_summaries"][0]["final_total"] = 5
    _, inventory, audit = reconcile_part(rows, source, labels)
    assert len(inventory) == 2
    assert audit["counts"]["active_total"] == 1
    assert audit["final_total_difference"] == -4
