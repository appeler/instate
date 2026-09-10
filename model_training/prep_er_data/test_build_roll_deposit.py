"""Small-state deposit contracts independent of the optional resolver install."""

from types import SimpleNamespace

import pyarrow as pa
from build_roll_deposit import canonical_input, count_recorded, prepare_records


def test_record_ids_preserve_serial_collisions_and_relationships():
    row = {
        "filename": "/raw/PART_1.pdf",
        "number": "12",
        "change": "",
        "relative_type": "Other",
    }
    rows = prepare_records([row, {**row, "change": "deleted"}], "edition")
    assert rows[0]["record_id"] != rows[1]["record_id"]
    assert rows[0]["relationship"] == "Other"
    assert not rows[0]["deleted"]
    assert rows[1]["deleted"]
    assert row["filename"] == "/raw/PART_1.pdf"


def test_relative_only_and_conflicting_candidates_do_not_enter_counts():
    candidates = pa.Table.from_pylist(
        [
            {"candidate_id": "own", "surname_latin_normalized": "S'Har-ma"},
            {"candidate_id": "relative", "surname_latin_normalized": "Sharma"},
        ]
    )
    resolutions = pa.Table.from_pylist(
        [
            {
                "record_id": "a",
                "recorded_candidate_id": "own",
                "selected_candidate_id": "own",
                "resolution_status": "recorded_selected",
            },
            {
                "record_id": "b",
                "recorded_candidate_id": None,
                "selected_candidate_id": "relative",
                "resolution_status": "relative_candidate",
            },
            {
                "record_id": "c",
                "recorded_candidate_id": "own",
                "selected_candidate_id": None,
                "resolution_status": "conflict",
            },
        ]
    )
    rows, counts = count_recorded(
        SimpleNamespace(candidates=candidates, resolutions=resolutions)
    )
    assert counts == {"sharma": 1}
    assert [row["training_eligible"] for row in rows] == [True, False, False]
    assert rows[1]["surname_latin_normalized"] == "Sharma"
    assert rows[1]["model_input"] is None


def test_canonical_input_matches_training_contract():
    assert canonical_input("S'Har-ma 42") == "sharma"
    assert canonical_input(None) == ""


def test_fallback_positions_are_state_specific():
    from build_roll_deposit import POSITION_POLICIES

    assert POSITION_POLICIES == {"andaman": "last", "dadra": "first"}
