# Coverage adjustment

`instate.coverage.adjust_surname_counts` is a separate, opt-in weighting module.
Surname resolution stays in `upnaam`; weighting does not rewrite source counts or
silently change training, runtime inference or packaged lookup tables.

## MCAR baseline

Assume surname observations are missing completely at random **within each state
and roll edition**. States may have different recovery rates. For each population:

```text
weight = target_count / observed_surname_count
estimated_count = observed_count * weight
```

Every observed surname in that population receives the same weight. For example,
15% recovery implies a weight of approximately 6.67. This is the chosen working
assumption, not a claim that missingness has been shown to be random.

```python
import pandas as pd
from instate.coverage import adjust_surname_counts

counts = pd.DataFrame(
    {
        "state": ["example", "example"],
        "year": [2017, 2017],
        "surname": ["patil", "shastri"],
        "observed_count": [30, 20],
    }
)
totals = pd.DataFrame(
    {"state": ["example"], "year": [2017], "target_count": [100]}
)
result = adjust_surname_counts(
    counts,
    totals,
    strata=["state", "year"],
    source_revision="observed-example-v1",
    target_revision="controls-example-v1",
    assumption="mcar",
)
print(result.estimates)
print(result.diagnostics)
```

The example produces weights of 2 and estimated counts of 60 and 40; observed
counts remain 30 and 20. `result.assumption` records `mcar`. The example numbers
are illustrative, not measured state recovery rates.

Choose the target according to what is being adjusted:

- **Surname non-resolution:** use the comparable parsed-elector total.
- **Overall corpus shortfall:** use the comparable electoral-roll total, also
  assuming missing source records are random within the state.

Using these elector totals assumes unresolved people have surname values drawn
from the same distribution as resolved people. Exclude independently known cases
where no family surname applies; an unresolved or one-word name alone does not
establish that status. No reference audit is required to run this assumption-based
baseline. Record the denominator's source, edition, geography and exclusions.

## Outputs and safeguards

`estimates` contains the grouping columns, surname, `observed_count`, `weight` and
`estimated_count`. `diagnostics` contains each group's `target_count`,
`observed_count`, `coverage_fraction`, `weight`, `unobserved_count` and `status`.
The result also retains the source revision, target revision and assumption.

Inputs require unique keys, positive integer observed surname counts and
nonnegative integer targets. Include observed counts before minimum-support
filtering. The function rejects invalid or unmatched inputs, observed totals above
targets, and positive-target groups with no observed surnames. Zero-target empty
groups have null weights and an `empty_stratum` status.

An entirely unobserved state cannot be weighted. Missing constituencies within an
observed state can be represented under the state-level MCAR assumption; that is
extrapolation, not a parser repair. Weights cannot introduce unseen surnames or
correct wrong surname assignments. Estimated mass must not count as additional
observed support for filtering or privacy thresholds. No uncertainty intervals
are currently calculated.

## Weighting plan

1. Produce state-by-edition MCAR estimates using explicit, comparable target totals.
   Keep the original counts and the adjusted output as separate artifacts.
2. Report coverage and weight for each state, alongside the chosen denominator.
   Do not call a larger estimated count a recovered source record.
3. With the Andaman final and Dadra draft frames rebuilt, continue with J&K,
   the missing Dadra final supplements, and Karnataka's missing constituencies.
   Recompute weights when observed counts change rather than adding recovered
   people to already expanded counts.
4. Use ration/land records later to improve surname recovery and assess the MCAR
   approximation. They provide person-level evidence, not extra electoral count
   mass. An audit is useful follow-up work, not a blocker for weighting.
5. If useful, compare finer geography or demographic strata with
   `assumption="conditional_mar"`. Both assumptions use the same expansion formula;
   the supplied strata determine which records share a recovery rate.
6. Evaluate weighted model training as a separate change. State multipliers preserve
   within-state surname proportions but can change count-derived state priors and
   `P(state | surname)`. Keep that choice explicit rather than automatically
   replacing existing model artifacts.

## Recovered roll benchmarks

For this analysis, printed electorate totals serve as the census-like adult-population
frame. The adjustment uses those edition-matched totals, not population projections.
Parser reconciliation and surname selection remain separate diagnostics.

| Roll frame | Printed electorate | Parsed frame records | Recorded surname selections | MCAR factor |
| --- | ---: | ---: | ---: | ---: |
| Andaman, final 2017 | 277,983 | 277,987 | 182,787 | 1.521 |
| Dadra, draft 2017 | 217,934 | 217,934 | 155,108 | 1.405 |

The separate estimates expand recorded surname selections to the printed frame.
They do not change the observed-count inputs to the lookup or model. Relative-only
surname candidates are retained in the resolution artifacts but excluded from
these observed counts.

Dadra's 266 main rolls form a complete draft frame. English final supplements are
missing for 11 parts, so the available final-roll records are retained separately,
not mixed into draft counts. Main PDFs contain later deletion annotations; the
reconstructed draft frame precedes those supplement events.

The reproducible deposit builder is `model_training/prep_er_data/build_roll_deposit.py`.
It requires the sibling `upnaam` source on `PYTHONPATH`, plus PyMuPDF and Pillow for
source-crop audits. Surname-first is the explicit fallback for Dadra; surname-last
is the fallback for Andaman. Relation evidence and conflict handling take precedence.
