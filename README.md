## instate: rank electoral-roll state and synthetic language targets from surnames

[![CI](https://github.com/appeler/instate/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/instate.svg)](https://pypi.org/project/instate)
[![Documentation](https://github.com/appeler/instate/actions/workflows/docs.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/docs.yml)
[![image](https://static.pepy.tech/badge/instate)](https://pepy.tech/project/instate)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/gojiberries/instate)

Instate reports how processed occurrences of a surname are distributed across
the included state records in 2017 Indian electoral rolls. Its models extend
that aggregate lookup by ranking state labels and a synthetic language target
for surnames outside the tables. The outputs do not estimate an individual's
residence, origin, or spoken language.

# What the outputs mean

The state lookup denominator is all included, processed occurrences of the
requested surname, not people in the current population. The state model learns
to rank the same state labels.

The language target is constructed rather than observed. For each state, the
builder assigns weights of 0.5, 0.25, 0.125, 0.0625, and 0.03125 to its five
ranked languages, then mixes those weights using the surname's electoral-roll
state distribution. It is neither a record of a person's language nor a list of
official languages.

# Dataset

The installed package bundles typed Parquet lookup tables for state and language
distributions. Parquet preserves the string, float, and count schemas used by
the public APIs without runtime CSV inference.

Refer to the
[notebooks](https://github.com/appeler/instate/tree/main/model_training/notebooks)
for the notebooks that were used to prepare the above datasets and train the
models.

# Web UI

The repository includes a Streamlit interface for CSV lookup and state
prediction:

```bash
uv sync --extra streamlit
uv run streamlit run streamlit/streamlit_app.py
```

# Installation

We strongly recommend installing instate inside a Python
virtual environment (see [venv
documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

    pip install instate

# API

instate provides four functions for surname lookup and label ranking.

## Electoral Rolls Lookup

- **get_state_distribution** - Get state shares among included, processed 2017
  electoral-roll surname occurrences

```python
import instate

# With list of names
names = ["sharma", "patel", "singh"]
result = instate.get_state_distribution(names)
print(result[["name", "Delhi", "Gujarat", "Punjab"]].head())

# With DataFrame
import pandas as pd

df = pd.DataFrame({"lastname": ["sharma", "patel"]})
result = instate.get_state_distribution(df, "lastname")
print(result.shape)  # (2, 36): name, total_n, and 34 state columns
```

> The bundled electoral lookup was rebuilt from the 2017 rolls and covers **all 34
> states/UTs**.
> Known-weak states from upstream romanization: **Telugu/Telangana** and **Gujarat**
> surnames are noisier (transliteration truncation / naming structure); other states are
> solid. Trailing-vowel spelling variants (e.g. Kannada `patila`, Odia `dasa`) are merged
> into their canonical forms (`patil`, `das`).

- **get_state_languages** - Map states to their official languages

```python
# Map states to languages
states = ["Delhi", "Punjab", "Gujarat"]
result = instate.get_state_languages(states)
print(result[["state", "official_languages"]])

#     state official_languages
# 0   Delhi     Hindi, English
# 1  Punjab            Punjabi
# 2 Gujarat           Gujarati
```

## Neural Network Predictions

- **predict_state** - Predict likely states using the character-BiLSTM model

```python
# Predict top 3 most likely states
names = ["sharma", "patel", "singh"]
result = instate.predict_state(names, top_k=3)
print(result["predicted_states"].iloc[0])
print(result["prediction_status"].iloc[0])
```

- **predict_language** - Predict likely languages using LSTM or k-nearest neighbor

```python
# LSTM neural network prediction (top 3)
result = instate.predict_language(names, model="lstm", top_k=3)
print(result["predicted_languages"].iloc[0])

# K-nearest neighbor lookup (single best)
result = instate.predict_language(names, model="knn")
print(result["predicted_languages"].iloc[0])
```

All three model paths accept romanized input using ASCII letters `a` through
`z` and require at least three supported characters. Prediction results include
`prediction_status`: `predicted`,
`predicted_unsupported_characters_removed`,
`abstained_empty_or_missing`, `abstained_too_short`, or
`abstained_unsupported_characters`. `instate.get_model_metadata()` returns the
supported alphabet and minimum length for `state:lstm`, `language:lstm`, and
`language:knn`.

The neural APIs return label rankings from raw model scores. The scores have not
been calibrated and the package does not expose them as probabilities.

## Complete Example

```python
import pandas as pd
import instate

# Sample data
df = pd.DataFrame({"person_id": [1, 2, 3], "lastname": ["sharma", "patel", "singh"]})

# Get state distributions from electoral rolls
state_dist = instate.get_state_distribution(df, "lastname")
print("Electoral rolls data shape:", state_dist.shape)

# Predict states with neural network
predicted_states = instate.predict_state(df, "lastname", top_k=3)
print("Top 3 predicted states:", predicted_states["predicted_states"].iloc[0])

# Predict languages
predicted_langs = instate.predict_language(df, "lastname", model="lstm", top_k=3)
print("Top 3 predicted languages:", predicted_langs["predicted_languages"].iloc[0])

# Map states to languages
states_df = pd.DataFrame({"state": ["Delhi", "Gujarat", "Punjab"]})
lang_map = instate.get_state_languages(states_df, "state")
print("State language mapping:")
print(lang_map[["state", "official_languages"]])
```

# Data

The underlying data for the package can be accessed at:
<https://doi.org/10.7910/DVN/ZXMVTJ>

# Evaluation

The state model is a 2-layer character-level **bidirectional LSTM**
([`model_training/train_state_lstm.py`](https://github.com/appeler/instate/blob/main/model_training/train_state_lstm.py)),
trained on the rebuilt 34-state v2 data. The **language** model
(`predict_language(model="lstm")`) uses the same character-level model family
and is trained on the synthetic language mixture derived from each surname's
state footprint.

The training programs canonicalize each surname to the exact lowercase ASCII
string consumed by the model, then assign those strings to deterministic,
disjoint train, validation, and test splits. Punctuation, spacing, case, and
digits cannot place equivalent model inputs in different splits. Training keeps
the earliest epoch with the best validation `mass_top3` and restores that epoch
before saving.

Training writes `<checkpoint>.training.json`. Untouched-test evaluation requires
that eligible manifest and verifies its data hash, checkpoint hash, seed, split
membership, source selection, and label order before loading the checkpoint.
Legacy, random, or mismatched checkpoints are refused. A matching checkpoint
can be evaluated with `--checkpoint <path> --evaluation-split test --eval-n 0`;
the result is written to `<checkpoint>.test-evaluation.json` by default.

Modal-label accuracy gives each surname one observation. Distribution-mass
coverage measures how much of the selected state or synthetic language target
falls inside the predicted labels. The checked-in
[`evaluation_manifest.json`](model_training/evaluation_manifest.json) records
the packaged reference-data and checkpoint hashes and the candidate membership
under the new contract. The original training files are not committed, and the
published checkpoints predate this split contract, so they are ineligible for
untouched-test labeling. Producing eligible metrics requires retraining under
the contract and then running the explicit test evaluation.

Metrics stay in run manifests rather than being copied into this README. The
neural checkpoints are downloaded from the
[versioned Hugging Face model repository](https://huggingface.co/gojiberries/instate) on first
use and cached by `huggingface-hub`. Set `INSTATE_MODEL_DIR` to a directory containing both
checkpoint files to use local artifacts instead.

# Authors

Atul Dhingra, Gaurav Sood and Rajashekar Chintalapati

# Contributor Code of Conduct

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the [Contributor Code of
Conduct](https://www.contributor-covenant.org/version/1/4/code-of-conduct/).

# License

The package is released under the [MIT
License](https://opensource.org/licenses/MIT).

## 🔗 Adjacent Repositories

- [appeler/naampy](https://github.com/appeler/naampy) — Infer Sociodemographic Characteristics from Names Using Indian Electoral Rolls
- [appeler/ethnicolr2](https://github.com/appeler/ethnicolr2) — Ethnicolr implementation with new models in pytorch
- [appeler/parsernaam](https://github.com/appeler/parsernaam) — AI name parsing. Predict first or last name using a DL model.
- [appeler/ethnicolor](https://github.com/appeler/ethnicolor) — Race and Ethnicity based on name using data from census, voter reg. files, etc.
- [appeler/ethnicolr](https://github.com/appeler/ethnicolr) — Predict Race and Ethnicity Based on the Sequence of Characters in a Name
