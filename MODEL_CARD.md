---
tags:
  - names
  - india
  - pytorch
---

# instate model artifacts

These local, unpublished 35-state artifacts power the estimation APIs in
[`instate`](https://github.com/appeler/instate). They require the matching
checkout and lookup table. Publication must pin a new immutable revision;
the previous 34-state checkpoint is incompatible.

## Files

| File | Package API | Output |
| --- | --- | --- |
| `instate_state_lstm.pt` | `instate.estimate_state_composition` | Calibrated state composition, 35 states and union territories |
| `instate_state_lstm_calibration.json` | same | Temperature, calibration objective, and before/after metrics |

The checkpoint is a PyTorch state dictionary for the two-layer
character-level bidirectional LSTM defined in `instate.nnets`. There is no
separate language model: `instate.estimate_language_composition` is a linear
mix of the state composition with Census 2011 mother-tongue shares shipped
inside the package, so it inherits this checkpoint's provenance.

## Target and training data

The model's softmax targets the distribution of a surname's processed
occurrences across the included electoral-roll records (2017 rolls, except
Assam's and Lakshadweep's 2026 rolls). The trainer
retains only surname-state cells with at least three records, then
samples pairs with probability proportional to retained record counts
and minimizes cross-entropy, whose minimizer is exactly that record-weighted
conditional distribution; the packaged lookup table reports the same
quantity for in-table surnames. Lookup totals sum only retained cells;
no suppressed cell contributes to their denominator. This target is not
residence or origin. The source data are available at
[parsed electoral-roll corpus](https://doi.org/10.7910/DVN/MUEGDT) and
[PDF corpus](https://doi.org/10.7910/DVN/OG47IV), and the complete
training programs are in the package repository under `model_training/`.

Surnames are canonicalized to the exact lowercase ASCII model input, then
assigned deterministically to disjoint 80% train, 10% validation, and 10%
test splits. Epoch selection uses the first 20,000 sorted validation names;
the remaining 165,007 form a separate calibration set. Training restores the
earliest epoch with the best selection-set `mass_top3` before saving.
Untouched-test evaluation requires the matching
eligible training manifest and validates the data, checkpoint, seed,
membership, source selection, and label order before evaluation.

## Evaluation

Untouched test split, 185,232 surnames weighted by 61.4 million records:

| metric | value |
| --- | --- |
| modal state accuracy, top 1 / top 3 | 0.508 / 0.764 |
| record mass covered, top 1 / top 3 | 0.469 / 0.751 |
| record-weighted log loss, calibrated | 1.724 |
| record-weighted Brier score, calibrated | 0.271 |
| top-1 confidence minus mass covered | -0.008 (0.074 before calibration) |

Modal-label accuracy gives each surname one observation and treats its most
frequent state as truth. Distribution-mass coverage weights labels by their
share of the surname's records. These are different estimands.

Calibration fits one temperature (1.263) on the separate calibration set by
minimizing record-weighted cross-entropy against each surname's empirical
state distribution; the calibration file records the objective and metrics.

## Loading

Install `instate` and use its public APIs. Direct loading requires the exact
architecture and label ordering from the same package version.

```python
import instate

states = instate.estimate_state_composition(["Singh", "Patel"])
languages = instate.estimate_language_composition(["Singh", "Patel"])
```

Set `INSTATE_MODEL_DIR` to a directory containing the artifacts to bypass
the Hub download in controlled or offline deployments.

Supported input is romanized ASCII `a` to `z` with at least three supported
characters; other inputs abstain with a machine-readable reason under the
appeler inference contract.

## Limitations

These outputs describe aggregate patterns in the training rolls. They do not
establish an individual's residence, origin, language, caste, ethnicity,
religion, or identity. Electoral-roll coverage, romanization, spelling,
shared surnames, and naming conventions can all produce systematic errors.
Roll coverage is uneven: Karnataka is at 15 percent of its electorate (five
northern districts), Jammu and Kashmir and Ladakh at 28 percent (Ladakh and
the Jammu region; the Urdu-only valley rolls are unparsed), Gujarat at 52
percent (OCR loss), and Assam and Lakshadweep come from 2026 rolls while other
states use 2017. Lakshadweep contributes 3,312 training occurrences after
surname selection and filtering, from 57,618 active parsed entries;
this is a selective sample. Across 38 Lakshadweep-bearing test surnames
(350 local record weight), the model never places Lakshadweep in its top
three. The added output label does not establish useful generalization for
Lakshadweep; its lookup has direct evidence for covered surnames.
Chhattisgarh is absent. A surname from an
under-covered state is pulled toward better-covered states that share it.
Gujarat names remain noisy from OCR; Telangana now comes from the English
2017 rolls. The per-state table is in the repository's
`model_training/prep_er_data/SOURCES.md`. The language composition
additionally assumes language and
surname are independent within a state, which understates
community-specific associations. Do not use these outputs for decisions
about a person or access to services.

## Licensing

The `instate` source code is MIT licensed. Consult the source dataset terms
and your intended use before redistributing or deploying the learned
weights.
