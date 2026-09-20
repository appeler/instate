"""Test calibrated selection-token transliteration decisions."""

from model_training.prep_er_data.jk_urdu_transliteration import (
    adjudicated_reading,
    majority_reading,
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
