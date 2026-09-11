"""Coverage adjustment conserves control totals without altering observed evidence."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from instate.coverage import adjust_surname_counts


def adjust(counts, totals, assumption="conditional_mar", **kwargs):
    return adjust_surname_counts(
        counts,
        totals,
        strata=["district"],
        source_revision="counts-v1",
        target_revision="controls-v1",
        assumption=assumption,
        **kwargs,
    )


@pytest.fixture
def counts():
    return pd.DataFrame(
        {
            "district": ["a", "a", "b"],
            "surname": ["patil", "shastri", "rai"],
            "observed_count": [30, 20, 25],
        }
    )


@pytest.fixture
def totals():
    return pd.DataFrame({"district": ["a", "b"], "target_count": [100, 100]})


@pytest.mark.parametrize("assumption", ["mcar", "conditional_mar"])
def test_stratum_weights_conserve_totals_and_preserve_observations(
    counts, totals, assumption
):
    original_counts, original_totals = counts.copy(deep=True), totals.copy(deep=True)
    result = adjust(counts, totals, assumption=assumption)
    assert result.assumption == assumption
    assert result.estimates["weight"].tolist() == [2, 2, 4]
    assert result.estimates["estimated_count"].tolist() == [60, 40, 100]
    estimated = result.estimates.groupby("district")["estimated_count"].sum()
    assert estimated.to_dict() == {"a": 100, "b": 100}
    assert result.diagnostics["coverage_fraction"].tolist() == [0.5, 0.25]
    assert_frame_equal(counts, original_counts)
    assert_frame_equal(totals, original_totals)


def test_whole_missing_stratum_is_not_blowed_up(counts, totals):
    missing = pd.concat(
        [totals, pd.DataFrame({"district": ["missing"], "target_count": [100]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="no observed surnames"):
        adjust(counts, missing)


def test_empty_zero_target_stratum_has_no_fabricated_weight(counts, totals):
    controls = pd.concat(
        [totals, pd.DataFrame({"district": ["empty"], "target_count": [0]})],
        ignore_index=True,
    )
    result = adjust(counts, controls)
    empty = result.diagnostics.set_index("district").loc["empty"]
    assert pd.isna(empty["weight"])
    assert empty["status"] == "empty_stratum"
    assert len(result.estimates) == len(counts)


def test_fully_observed_stratum_has_unit_weight(counts):
    totals = pd.DataFrame({"district": ["a", "b"], "target_count": [50, 25]})
    result = adjust(counts, totals)
    assert result.estimates["weight"].eq(1).all()
    assert result.diagnostics["status"].eq("fully_observed").all()


@pytest.mark.parametrize("bad", [-1, 1.5, np.nan, np.inf, 2**54])
def test_invalid_counts_are_rejected(counts, totals, bad):
    counts = counts.astype({"observed_count": float})
    counts.loc[0, "observed_count"] = bad
    with pytest.raises(ValueError, match="finite integers"):
        adjust(counts, totals)


@pytest.mark.parametrize("bad", ["30", True])
def test_strings_and_booleans_are_not_counts(counts, totals, bad):
    with pytest.raises(ValueError, match="numeric"):
        adjust(counts.assign(observed_count=bad), totals)


def test_duplicate_missing_and_unmatched_keys(counts, totals):
    with pytest.raises(ValueError, match="unique"):
        adjust(pd.concat([counts, counts]), totals)
    with pytest.raises(ValueError, match="unique"):
        adjust(counts, pd.concat([totals, totals]))
    with pytest.raises(ValueError, match="missing"):
        adjust(counts.assign(surname=None), totals)
    with pytest.raises(ValueError, match="empty strings"):
        adjust(counts.assign(surname=[" ", "shastri", "rai"]), totals)
    with pytest.raises(ValueError, match="control total"):
        adjust(counts, totals.iloc[:1])
    with pytest.raises(ValueError, match="exceed"):
        adjust(counts, totals.assign(target_count=1))


def test_schema_and_assumptions_are_explicit(counts, totals):
    with pytest.raises(ValueError, match="columns"):
        adjust(counts.assign(weight=1), totals)
    with pytest.raises(ValueError, match="surnames"):
        adjust(counts.assign(surname=[1, 2, 3]), totals)
    for strata in ([], ["district", "district"], ["surname"], [""]):
        with pytest.raises(ValueError, match=r"strata|stratum"):
            adjust_surname_counts(
                counts,
                totals,
                strata=strata,
                source_revision="v1",
                target_revision="v1",
                assumption="conditional_mar",
            )
    with pytest.raises(ValueError, match="revisions"):
        adjust_surname_counts(
            counts,
            totals,
            strata=["district"],
            source_revision="",
            target_revision="v1",
            assumption="conditional_mar",
        )
    with pytest.raises(ValueError, match="assumption"):
        adjust_surname_counts(
            counts,
            totals,
            strata=["district"],
            source_revision="v1",
            target_revision="v1",
            assumption="unverified",
        )


@pytest.mark.parametrize("assumption", ["mcar", "conditional_mar"])
def test_multiple_strata_do_not_pool_different_years(assumption):
    counts = pd.DataFrame(
        {
            "state": ["a", "a"],
            "year": [2017, 2018],
            "surname": ["patil", "patil"],
            "observed_count": [25, 50],
        }
    )
    totals = pd.DataFrame(
        {"state": ["a", "a"], "year": [2017, 2018], "target_count": [100, 100]}
    )
    result = adjust_surname_counts(
        counts,
        totals,
        strata=["state", "year"],
        source_revision="v1",
        target_revision="v1",
        assumption=assumption,
    )
    assert result.estimates["weight"].tolist() == [4, 2]
