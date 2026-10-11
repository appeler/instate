"""Test calibrated selection-token transliteration decisions."""

from model_training.prep_er_data.jk_urdu_transliteration import (
    adjudicated_reading,
    majority_reading,
    unflagged_reading,
)


def decision(latin, *, status="supported", gate=True):
    return {"latin": latin, "status": status, "batch_gate_passed": gate}


def test_majority_reading_requires_two_supported_passing_votes():
    rows = [decision("Dar"), decision("dar"), decision("Daar")]
    assert majority_reading(rows) == "dar"
    assert majority_reading([decision("Dar"), decision("Daar")]) is None
    assert majority_reading([decision("Dar"), decision("Dar", gate=False)]) is None
    rows = [decision("Dar"), decision("Dar", status="uncertain")]
    assert majority_reading(rows) is None


def test_majority_reading_rejects_tie():
    assert (
        majority_reading(
            [decision("Dar"), decision("Dar"), decision("Daar"), decision("Daar")]
        )
        is None
    )


def test_adjudicated_reading_uses_prior_to_break_persistent_tie():
    rows = [decision("Baqia"), decision("Baqiya")] * 2
    assert adjudicated_reading(rows, "Baqia") == (
        "baqia",
        "selected_tie_prior_primary",
    )
    assert adjudicated_reading(rows, "Baqiyah") == (
        "baqiya",
        "selected_tie_prior_similarity",
    )


def clean_reviews():
    return [
        {
            **decision(latin),
            "pass": name,
            "source_warning": False,
            "attested_forms": ["dar"],
        }
        for name, latin in zip(("a", "b", "c"), ("Dar", "dar", "Daar"), strict=True)
    ]


def test_unflagged_reading_accepts_repeated_attested_primary():
    assert unflagged_reading(clean_reviews(), None) == (
        "dar",
        "selected_unflagged_majority",
    )


def test_unflagged_reading_preserves_historical_source_warning():
    assert unflagged_reading(clean_reviews(), {"source_warning": True}) == (
        None,
        "prior_source_warning",
    )


def test_unflagged_reading_requires_three_distinct_clean_passes():
    rows = clean_reviews()
    assert unflagged_reading(rows[:2], None)[0] is None
    assert unflagged_reading([rows[0], rows[0], rows[2]], None)[0] is None
    for field, value in (
        ("source_warning", True),
        ("status", "uncertain"),
        ("batch_gate_passed", False),
        ("latin", "two words"),
    ):
        rows = clean_reviews()
        rows[2][field] = value
        assert unflagged_reading(rows, None)[0] is None


def test_unflagged_reading_abstains_on_reference_or_primary_disagreement():
    rows = clean_reviews()
    for row in rows:
        row["attested_forms"] = ["daar"]
    assert unflagged_reading(rows, None) == (None, "reference_disagrees")
    rows = clean_reviews()
    rows[1]["latin"] = "darh"
    assert unflagged_reading(rows, None) == (None, "no_repeated_primary")
