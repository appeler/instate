---
tags:
  - names
  - india
  - pytorch
---

# instate model artifacts

These artifacts power the estimation APIs in
[`instate`](https://github.com/appeler/instate). The package downloads this
repository at an immutable commit so a released package cannot silently
change models.

## Files

| File | Package API | Output |
| --- | --- | --- |
| `instate_state_lstm.pt` | `instate.estimate_state_composition` | Calibrated state composition, 34 states and union territories |
| `instate_state_lstm_calibration.json` | same | Temperature, calibration objective, and before/after metrics |

The checkpoint is a PyTorch state dictionary for the two-layer
character-level bidirectional LSTM defined in `instate.nnets`. There is no
separate language model: `instate.estimate_language_composition` is a linear
mix of the state composition with Census 2011 mother-tongue shares shipped
inside the package, so it inherits this checkpoint's provenance.

## Target and training data

The model's softmax targets the distribution of a surname's processed
occurrences across the included 2017 electoral-roll records. The trainer
samples surname-state pairs with probability proportional to record counts
and minimizes cross-entropy, whose minimizer is exactly that record-weighted
conditional distribution; the packaged lookup table reports the same
quantity for in-table surnames. This target is not residence or origin. The
source data are available at
[Harvard Dataverse](https://doi.org/10.7910/DVN/ZXMVTJ), and the complete
training programs are in the package repository under `model_training/`.

Surnames are canonicalized to the exact lowercase ASCII model input, then
assigned deterministically to disjoint 80% train, 10% validation, and 10%
test splits. Training restores the earliest epoch with the best validation
`mass_top3` before saving. Untouched-test evaluation requires the matching
eligible training manifest and validates the data, checkpoint, seed,
membership, source selection, and label order before evaluation.

## Evaluation

Untouched test split, 177,019 surnames weighted by 58.3 million records:

| metric | value |
| --- | --- |
| modal state accuracy, top 1 / top 3 | 0.534 / 0.770 |
| record mass covered, top 1 / top 3 | 0.447 / 0.668 |
| record-weighted log loss, calibrated | 1.762 |
| record-weighted Brier score, calibrated | 0.284 |
| top-1 confidence minus mass covered | 0.040 (0.106 before calibration) |

Modal-label accuracy gives each surname one observation and treats its most
frequent state as truth. Distribution-mass coverage weights labels by their
share of the surname's records. These are different estimands.

Calibration fits one temperature (1.207) on the validation split by
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
shared surnames, and naming conventions can all produce systematic errors;
Telugu, Telangana, and Gujarat names are known to be especially noisy in the
source pipeline. The language composition additionally assumes language and
surname are independent within a state, which understates
community-specific associations. Do not use these outputs for decisions
about a person or access to services.

## Licensing

The `instate` source code is MIT licensed. Consult the source dataset terms
and your intended use before redistributing or deploying the learned
weights.
