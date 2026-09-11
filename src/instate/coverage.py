"""Explicit stratum-based coverage adjustment, separate from surname inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


@dataclass(frozen=True)
class CoverageAdjustment:
    """Observed counts, estimated counts and the assumptions behind their difference."""

    estimates: pd.DataFrame
    diagnostics: pd.DataFrame
    source_revision: str
    target_revision: str
    assumption: str


def _validate_table(frame: pd.DataFrame, columns: set[str], keys: list[str]) -> None:
    if not frame.columns.is_unique or set(frame.columns) != columns:
        raise ValueError(f"table columns must be exactly {sorted(columns)}")
    if frame[keys].isna().to_numpy().any():
        raise ValueError("stratum and surname keys cannot be missing")
    if frame.duplicated(keys).any():
        raise ValueError("table keys must be unique")
    for key in keys:
        if (
            cast("pd.Series", frame[key])
            .map(lambda value: isinstance(value, str) and not value.strip())
            .any()
        ):
            raise ValueError("stratum and surname keys cannot be empty strings")


def _validate_counts(values: pd.Series, *, positive: bool) -> None:
    if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
        raise ValueError("counts must be numeric, not strings or booleans")
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    if (
        not np.isfinite(numeric).all()
        or (numeric < (1 if positive else 0)).any()
        or (numeric != np.floor(numeric)).any()
        or (numeric > 2**53).any()
    ):
        raise ValueError("counts must be finite integers within the supported range")


def adjust_surname_counts(
    counts: pd.DataFrame,
    totals: pd.DataFrame,
    *,
    strata: Sequence[str],
    source_revision: str,
    target_revision: str,
    assumption: Literal["mcar", "conditional_mar"],
) -> CoverageAdjustment:
    """Estimate surname counts by reweighting within explicitly supplied strata.

    Args:
        counts: Unique stratum/surname rows with positive integer observed_count.
            Include all observed surnames before minimum-cell suppression.
        totals: Unique stratum rows with nonnegative integer target_count, using
            matching source geography, year, population and surname definition.
        strata: Shared grouping columns, such as state, year, district and sex.
        source_revision: Immutable revision of the observed surname counts.
        target_revision: Immutable revision of the population control totals.
        assumption: Explicit mcar assumption within each supplied population,
            typically state and roll edition, or conditional_mar for adjustment
            within finer covariate strata. Neither assumption is tested here.

    Returns:
        Separate observed and estimated counts, weights and coverage diagnostics.
        The inputs are unchanged. Zero-target empty strata have null weights.
        No weights are applied to inference, training or packaged lookup tables.

    Raises:
        ValueError: Schemas, counts, support, revisions or assumptions are invalid.

    Notes:
        Weighting does not correct misclassified surnames or identify unobserved
        surname categories. Known absence of a family surname is not a missing
        surname to impute. Entirely missing positive-target strata cannot be
        recovered by weighting. This function cannot verify missingness assumptions.
    """
    keys = list(strata)
    reserved = {
        "surname",
        "observed_count",
        "target_count",
        "weight",
        "estimated_count",
        "coverage_fraction",
        "unobserved_count",
        "status",
    }
    if not keys or len(set(keys)) != len(keys) or reserved.intersection(keys):
        raise ValueError("strata must be distinct nonempty, nonreserved column names")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("stratum column names must be nonempty strings")
    if not source_revision.strip() or not target_revision.strip():
        raise ValueError("source and target revisions must be nonempty")
    if assumption not in {"mcar", "conditional_mar"}:
        raise ValueError("the mcar or conditional_mar assumption must be explicit")
    _validate_table(counts, {*keys, "surname", "observed_count"}, [*keys, "surname"])
    _validate_table(totals, {*keys, "target_count"}, keys)
    if (
        not cast("pd.Series", counts["surname"])
        .map(lambda value: isinstance(value, str))
        .all()
    ):
        raise ValueError("surnames must be nonempty strings")
    _validate_counts(cast("pd.Series", counts["observed_count"]), positive=True)
    _validate_counts(cast("pd.Series", totals["target_count"]), positive=False)
    represented = counts.merge(
        totals[keys], on=keys, how="left", validate="many_to_one", indicator=True
    )
    if (represented["_merge"] != "both").any():
        raise ValueError("every observed stratum must have a control total")
    observed = (
        counts.assign(observed_count=counts["observed_count"].astype(float))
        .groupby(keys, as_index=False, observed=True)["observed_count"]
        .sum()
    )
    observed = cast("pd.DataFrame", observed)
    diagnostics = totals.merge(observed, on=keys, how="left", validate="one_to_one")
    diagnostics["observed_count"] = diagnostics["observed_count"].fillna(0)
    if (diagnostics["observed_count"] > diagnostics["target_count"]).any():
        raise ValueError(
            "observed counts exceed target totals; reconcile definitions or duplicates"
        )
    unsupported = (diagnostics["target_count"] > 0) & (
        diagnostics["observed_count"] == 0
    )
    if unsupported.any():
        raise ValueError(
            "positive-target strata with no observed surnames cannot be weighted"
        )
    diagnostics["coverage_fraction"] = diagnostics["observed_count"] / diagnostics[
        "target_count"
    ].where(diagnostics["target_count"] > 0)
    diagnostics["weight"] = diagnostics["target_count"] / diagnostics[
        "observed_count"
    ].where(diagnostics["observed_count"] > 0)
    diagnostics["unobserved_count"] = (
        diagnostics["target_count"] - diagnostics["observed_count"]
    )
    diagnostics["status"] = np.where(
        diagnostics["target_count"] == 0,
        "empty_stratum",
        np.where(diagnostics["coverage_fraction"] == 1, "fully_observed", "adjusted"),
    )
    estimates = counts.merge(
        diagnostics[[*keys, "weight"]], on=keys, how="left", validate="many_to_one"
    )
    estimates["estimated_count"] = estimates["observed_count"] * estimates["weight"]
    return CoverageAdjustment(
        estimates, diagnostics, source_revision, target_revision, assumption
    )
