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

Both files are PyTorch state dictionaries for the two-layer character-level
bidirectional LSTM defined in `instate.nnets`.

## Training data and evaluation

The state checkpoint was trained on surname counts rebuilt from Indian
electoral rolls covering 34 states and union territories. The language target
distributions were derived from each surname's state distribution and the
package's state-to-official-language mapping. The source data are available at
[Harvard Dataverse](https://doi.org/10.7910/DVN/ZXMVTJ), and the complete
training programs are in the package repository under `model_training/`.

The bundled training programs use a seeded 80/20 surname split. On the complete
held-out split, the published state checkpoint produced 54.6% modal-label top-1
accuracy, 78.0% modal-label top-3 accuracy, 44.6% distribution-mass top-1
coverage, and 69.4% distribution-mass top-3 coverage. The language checkpoint
produced 58.9%, 78.8%, 49.0%, and 76.1% on the same four measures.

Modal-label accuracy gives each held-out surname one observation and treats its
most frequent label as truth. Distribution-mass coverage weights each label by
its observed count or probability for that surname. These are different
estimands and should not be compared as if they were the same accuracy measure.

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

## Limitations

These models estimate aggregate patterns in the training rolls. They do not
establish an individual's residence, language, caste, ethnicity, religion, or
identity. Electoral-roll coverage, romanization, spelling, migration, shared
surnames, and naming conventions can all produce systematic errors. Telugu,
Telangana, and Gujarat names are known to be especially noisy in the source
pipeline. Do not use these outputs for decisions about a person or access to
services.

## Licensing

The `instate` source code is MIT licensed. Consult the source dataset terms and
your intended use before redistributing or deploying the learned weights.
