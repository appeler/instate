---
tags:
  - names
  - india
  - pytorch
---

# instate model artifacts

These 35-state artifacts power the estimation APIs in
[`instate`](https://github.com/appeler/instate) 3.2.0. The package pins an
immutable Hugging Face revision, and its manifests record SHA-256 hashes.

## Files

| File | Package API | Output |
| --- | --- | --- |
| `instate_state_lstm.safetensors` | `instate.estimate_state_composition` | Calibrated state composition, 35 states and union territories |
| `instate_state_lstm_calibration.json` | same | Temperature, calibration objective, and before/after metrics |
| `instate_unique_ln_state_prop_v2.parquet` | `instate.lookup_state_composition` | Retained surname counts and state shares |

The checkpoint stores SafeTensors weights for the two-layer
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

## Evaluation

The state model is a two-layer character-level bidirectional LSTM trained on
1,483,554 canonical names. Hash assignment fixes the train, validation, and
test memberships. A separate hash orders validation names: the first 20,000
choose the epoch with the lowest record-weighted cross-entropy, and the other
165,724 fit one calibration temperature (1.200). Training restored
epoch 7 after eight epochs.

These uncalibrated scores use the 20,000 names that chose the checkpoint
(4,527,362 retained records). They are development evidence and do not
establish generalization. The historical test had already informed development;
it was not rescored, and this candidate cannot claim an untouched test result.

| Checkpoint | Log loss | Brier score | Top-three record mass |
| --- | ---: | ---: | ---: |
| Released 3.1 | 1.473 | 0.276 | 79.2% |
| Earlier initials candidate | 1.553 | 0.288 | 78.4% |
| Corrected selection | 1.374 | 0.235 | 79.2% |

Lower log loss and Brier score indicate closer agreement with the retained
state distributions. Top-three coverage is the share of record mass assigned
to those states. These measures weight source records, not people.

The checkpoint, data, split memberships, and calibration are bound by hashes.
The training and comparison records are in `model_training/karnataka_2017_training.json`
and `model_training/karnataka_2017_model_diagnostic.json`. Earlier test results
remain under `model_training/history/` and describe earlier checkpoints.

Model artifacts download automatically from a pinned Hugging Face revision.
For offline use, set `INSTATE_MODEL_DIR` to a directory containing
`instate_state_lstm.safetensors`, the matching calibration JSON, and lookup Parquet.

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
religion, or identity. Electoral-roll coverage, romanization, spelling, shared
surnames, and naming conventions can all produce systematic errors. Karnataka
covers 196 of 224 source constituencies, with all 28 Bengaluru constituencies
absent. Surname selection covers 44.3% of its 40,389,176 active parsed records.
Its initials fallbacks retain the sole usable name word without establishing
that it is a hereditary surname. Jammu and Kashmir and Ladakh cover 28% of the
2019 electorate (Ladakh and the Jammu region; the Urdu-only valley rolls are
unparsed), and Gujarat covers 52% because of OCR loss. Assam and Lakshadweep
use 2026 rolls; the other states use 2017 rolls.

Lakshadweep contributes 3,312 training occurrences after surname selection and
filtering, from 57,618 active parsed entries. This is a selective sample.
The current selection sample has only four Lakshadweep-bearing names. The
lookup supplies direct evidence for covered surnames. Chhattisgarh is absent.
A surname from a poorly covered state is pulled toward better-covered states
that share it. Gujarat
names remain noisy from OCR; Telangana now comes from the English 2017 rolls.
The per-state table is in the repository's
`model_training/prep_er_data/SOURCES.md`. The language composition additionally
assumes language and surname are independent within a state, which understates
community-specific associations. Do not use these outputs for decisions about a
person or access to services.

## Licensing

The `instate` source code is MIT licensed. Consult the source dataset terms
and your intended use before redistributing or deploying the learned
weights.
