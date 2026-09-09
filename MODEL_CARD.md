---
tags:
  - names
  - india
  - pytorch
---

# instate model artifacts

These 35-state artifacts power the estimation APIs in
[`instate`](https://github.com/appeler/instate) 3.2.0. The package pins an
immutable Hub revision and verifies each runtime artifact by SHA-256.

## Files

| File | Package API | Output |
| --- | --- | --- |
| `instate_state_lstm.pt` | `instate.estimate_state_composition` | Calibrated state composition, 35 states and union territories |
| `instate_state_lstm_calibration.json` | same | Temperature, calibration objective, and before/after metrics |
| `instate_unique_ln_state_prop_v2.parquet` | `instate.lookup_state_composition` | Retained surname counts and state shares |

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
the remaining 164,480 form a separate calibration set. Training restores the
earliest epoch with the best selection-set `mass_top3` before saving.
Untouched-test evaluation requires the matching
eligible training manifest and validates the data, checkpoint, seed,
membership, source selection, and label order before evaluation.

## Evaluation

Untouched test split, 184,752 surnames weighted by 61.7 million records:

| metric | value |
| --- | --- |
| modal state accuracy, top 1 / top 3 | 0.500 / 0.754 |
| record mass covered, top 1 / top 3 | 0.473 / 0.667 |
| record-weighted log loss, calibrated | 1.820 |
| record-weighted Brier score, calibrated | 0.292 |
| top-1 confidence minus mass covered | 0.005 (0.067 before calibration) |

Modal-label accuracy gives each surname one observation and treats its most
frequent state as truth. Distribution-mass coverage weights labels by their
share of the surname's records. These are different estimands.

Calibration fits one temperature (1.186) on the separate calibration set by
minimizing record-weighted cross-entropy against each surname's empirical
state distribution; the calibration file records the objective and metrics.

On the same new test names and retained-count targets, the 3.1 checkpoint has
record-weighted log loss 1.728 and top-three mass coverage 0.750, compared with
1.820 and 0.667 for this checkpoint. Karnataka's local top-one record coverage
rises from 37.6% to 44.3%, while its top-three coverage changes from 59.4% to
58.9%. National predictive performance is lower for this checkpoint. Full
comparisons are in `model_training/karnataka_2017_model_diagnostic.json`.

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
Karnataka covers 196 of 224 source constituencies, with all 28 Bengaluru
constituencies absent. Surname selection covers 28.6% of its 40,389,176 active
parsed records. Jammu and Kashmir and Ladakh cover 28% of the 2019 electorate
(Ladakh and the Jammu region; the Urdu-only valley rolls are unparsed), and
Gujarat covers 52% because of OCR loss. Assam and Lakshadweep use 2026 rolls;
the other states use 2017 rolls.

Lakshadweep contributes 3,312 training occurrences after surname selection
and filtering, from 57,618 active parsed entries. This is a selective sample.
Across 38 Lakshadweep-bearing test surnames (350 local record weight), the
model never places Lakshadweep in its top three. The lookup supplies direct
evidence for covered surnames. Chhattisgarh is absent. A surname from an
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
