---
tags:
  - names
  - india
  - pytorch
---

# instate model artifacts

These checkpoints power the neural prediction APIs in
[`instate`](https://github.com/appeler/instate). The package downloads this
repository at an immutable commit so a released package cannot silently change
models.

## Files

| File | Package API | Output |
| --- | --- | --- |
| `instate_state_lstm.pt` | `instate.predict_state` | Ranked Indian states and union territories |
| `instate_lang_lstm.pt` | `instate.predict_language(..., model="lstm")` | Ranked languages |

Both files are PyTorch state dictionaries for the character-level bidirectional
LSTM defined in `instate.nnets`. The state model has two layers; the language
model has one.

## Training data and evaluation

The state checkpoint targets the distribution of processed surname occurrences
across the included 2017 electoral-roll records for 34 states and union
territories. This target is not residence or origin. The language target is a
synthetic mixture: each state's five ranked languages receive geometric weights
of 0.5, 0.25, 0.125, 0.0625, and 0.03125, and the surname's state distribution
mixes the state vectors. It is not an observed or official-language label. The
source data are available at
[Harvard Dataverse](https://doi.org/10.7910/DVN/ZXMVTJ), and the complete
training programs are in the package repository under `model_training/`.

The training programs now assign surnames deterministically to disjoint 80%
train, 10% validation, and 10% test splits. Training uses validation metrics;
test evaluation requires an explicit run against a saved checkpoint. Run
manifests record data and checkpoint hashes, label order, split membership
hashes, evaluated membership, and computed metrics.

The published checkpoints predate this evaluation contract. The repository's
`model_training/evaluation_manifest.json` binds the packaged reference tables
and current model artifacts and records candidate membership under the new
contract. The original training files are not committed, and the manifest does
not assert untouched-test metrics. New contract-compliant metrics require
retraining and explicit test evaluation.

Modal-label accuracy gives each evaluated surname one observation and treats its
most frequent label as truth. Distribution-mass coverage weights each label by
its state-occurrence or synthetic target mass for that surname. These are
different estimands and should not be compared as if they were the same
accuracy measure.

## Loading

Install `instate` and use its public APIs. Direct loading requires the exact
architecture and label ordering from the same package version.

```python
import instate

states = instate.predict_state(["Singh", "Patel"], top_k=3)
languages = instate.predict_language(["Singh", "Patel"], top_k=3)
```

Set `INSTATE_MODEL_DIR` to a directory containing both files to bypass the Hub
download in controlled or offline deployments.

The models and KNN lookup support romanized surnames containing ASCII `a` to
`z` and require at least three supported characters. Public prediction results
include a `prediction_status` reason, and `instate.get_model_metadata()` returns
the supported alphabet for each model path. Neural outputs are rankings from
raw scores, not calibrated probabilities.

## Limitations

These models rank aggregate targets constructed from the training rolls. They
do not establish an individual's residence, origin, language, caste, ethnicity,
religion, or identity. Electoral-roll coverage, romanization, spelling, shared
surnames, and naming conventions can all produce systematic errors. Telugu,
Telangana, and Gujarat names are known to be especially noisy in the source
pipeline. Do not use these outputs for decisions about a person or access to
services.

## Licensing

The `instate` source code is MIT licensed. Consult the source dataset terms and
your intended use before redistributing or deploying the learned weights.
