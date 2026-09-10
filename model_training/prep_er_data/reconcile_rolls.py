"""Reconcile part-scoped electoral-roll events without dropping their audit trail."""

from collections import defaultdict
from pathlib import Path


def reconcile_events(main_rows, events):
    """Return retained elector rows and a ledger for every supplement event.

    Match a unique EPIC within a part, then a unique serial when no conflicting
    EPIC is printed. Unmatched, ambiguous, and conflicting events are retained
    in the ledger, never applied by arbitrary first-match selection.
    """
    records = []
    by_epic, by_serial = defaultdict(list), defaultdict(list)

    def add(row):
        record = dict(row)
        record["record_key"] = (
            f"{Path(row['filename']).name}:{row['number']}:{len(records) + 1}"
        )
        record["deleted"] = row.get("change") == "deleted"
        records.append(record)
        by_serial[(row["part_no"], row["number"].strip())].append(record)
        epic = row.get("id", "").strip()
        if epic:
            by_epic[(row["part_no"], epic)].append(record)
        return record

    for row in main_rows:
        if row.get("change"):
            raise ValueError("Main input must be an unreconciled base roll")
        add(row)
    ledger = []
    for ordinal, event in enumerate(events, 1):
        kind = event.get("change")
        if kind not in {"added", "deleted", "corrected"}:
            raise ValueError(f"Unknown supplement event: {kind!r}")
        part = event["part_no"]
        epic = event.get("id", "").strip()
        entry = {
            **event,
            "event_key": f"{Path(event['filename']).name}:{ordinal}",
            "event_status": None,
            "match_method": None,
            "matched_record_key": None,
        }
        matches = by_epic[(part, epic)] if epic else []
        method = "epic"
        if len(matches) > 1:
            intersection = [
                record
                for record in matches
                if record["number"].strip() == event["number"].strip()
            ]
            if len(intersection) == 1:
                matches = intersection
                method = "epic_and_serial"
        if not matches:
            matches = by_serial[(part, event["number"].strip())]
            method = "serial"
        if len(matches) > 1:
            entry["event_status"] = "ambiguous"
        elif not matches:
            if kind == "added":
                matched = add(event)
                entry.update(
                    event_status="applied",
                    matched_record_key=matched["record_key"],
                    match_method="new_record",
                )
            else:
                entry["event_status"] = "unmatched"
        else:
            matched = matches[0]
            entry.update(matched_record_key=matched["record_key"], match_method=method)
            printed_id = matched.get("id", "").strip()
            if epic and printed_id and epic != printed_id:
                entry["event_status"] = "identifier_conflict"
            elif kind == "added":
                fields = (
                    "elector_name",
                    "father_or_husband_name",
                    "house_no",
                    "age",
                    "sex",
                )
                same = all(
                    event.get(k, "").strip().casefold()
                    == matched.get(k, "").strip().casefold()
                    for k in fields
                )
                entry["event_status"] = (
                    "duplicate"
                    if same and not matched["deleted"]
                    else "addition_conflict"
                )
            elif kind == "deleted":
                if matched["deleted"]:
                    entry["event_status"] = "duplicate"
                else:
                    matched["deleted"] = True
                    matched["change"] = "deleted"
                    entry["event_status"] = "applied"
            elif matched["deleted"]:
                entry["event_status"] = "correction_of_deleted_record"
            else:
                for field in (
                    "elector_name",
                    "father_or_husband_name",
                    "relative_type",
                    "has_husband",
                    "house_no",
                    "age",
                    "sex",
                ):
                    if field in event:
                        matched[field] = event[field]
                matched["change"] = "corrected"
                entry["event_status"] = "applied"
        ledger.append(entry)
    return records, ledger
