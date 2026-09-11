"""Supplement reconciliation guards, including repeated deletion events."""

from reconcile_rolls import reconcile_events


def row(part="1", serial="1", epic="ABC123", change="", name="TEST NAME"):
    return {
        "filename": f"{part}_{change or 'main'}.pdf",
        "part_no": part,
        "number": serial,
        "id": epic,
        "change": change,
        "elector_name": name,
        "age": "30",
    }


def test_duplicate_deletion_keeps_both_events_but_deletes_once():
    main = [row()]
    event = row(change="deleted")
    records, ledger = reconcile_events(main, [event, event])
    assert len(records) == 1
    assert records[0]["deleted"]
    assert [r["event_status"] for r in ledger] == ["applied", "duplicate"]
    assert main[0]["change"] == ""


def test_matching_is_scoped_to_part():
    records, ledger = reconcile_events(
        [row(part="1"), row(part="2")], [row(part="2", change="deleted")]
    )
    assert [r["deleted"] for r in records] == [False, True]
    assert ledger[0]["match_method"] == "epic"


def test_conflicting_id_does_not_delete_by_serial():
    records, ledger = reconcile_events(
        [row()], [row(epic="DIFFERENT", change="deleted")]
    )
    assert not records[0]["deleted"]
    assert ledger[0]["event_status"] == "identifier_conflict"


def test_absent_epic_can_match_unique_serial():
    records, ledger = reconcile_events([row()], [row(epic="", change="deleted")])
    assert records[0]["deleted"]
    assert ledger[0]["match_method"] == "serial"


def test_ambiguous_serial_abstains():
    records, ledger = reconcile_events(
        [row(epic="A"), row(epic="B")], [row(epic="", change="deleted")]
    )
    assert not any(r["deleted"] for r in records)
    assert ledger[0]["event_status"] == "ambiguous"


def test_missing_record_is_not_created_by_deletion():
    records, ledger = reconcile_events([], [row(change="deleted")])
    assert records == []
    assert ledger[0]["event_status"] == "unmatched"


def test_serial_disambiguates_duplicated_epic():
    records, ledger = reconcile_events(
        [row(serial="10"), row(serial="11")],
        [row(serial="11", change="deleted")],
    )
    assert [record["deleted"] for record in records] == [False, True]
    assert ledger[0]["event_status"] == "applied"
    assert ledger[0]["match_method"] == "epic_and_serial"


def test_addition_then_correction_retains_record_identity_and_events():
    records, ledger = reconcile_events(
        [], [row(change="added"), row(change="corrected", name="CORRECTED NAME")]
    )
    assert len(records) == 1
    assert records[0]["elector_name"] == "CORRECTED NAME"
    assert ledger[0]["matched_record_key"] == ledger[1]["matched_record_key"]
