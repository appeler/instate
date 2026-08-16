## instate: predict spoken language and the state of residence from last name

[![CI](https://github.com/appeler/instate/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/instate.svg)](https://pypi.org/project/instate)
[![Documentation](https://github.com/appeler/instate/actions/workflows/docs.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/docs.yml)
[![image](https://static.pepy.tech/badge/instate)](https://pepy.tech/project/instate)

Using the Indian electoral rolls data (2017), we provide a Python
package that takes the last name of a person and gives its distribution
across states. This package can also predict the spoken language of the
person based on the last name.

# Potential Use Cases

India has 22 official languages. To serve such a diverse language base
is a challenge for businesses and surveyors. To the extent that
businesses have access to the last name (and no other information) and
in the absence of other data that allows us to model a person\'s spoken
language, the distribution of last names across states is the best we
have.

# Dataset

The installed package bundles `lastname_langs_india.csv.tar.gz`, which is used
to look up the likely spoken language from a last name.

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

instate provides 4 main functions for predicting state and language from Indian last names.

## Electoral Rolls Lookup

- **get_state_distribution** - Get P(state|lastname) from 2017 electoral rolls data

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

- **predict_state** - Predict likely states using the bundled character-BiLSTM model

```python
# Predict top 3 most likely states
names = ["sharma", "patel", "singh"]
result = instate.predict_state(names, top_k=3)
print(result["predicted_states"].iloc[0])
# ['Delhi', 'Uttar Pradesh', 'Bihar']
```

- **predict_language** - Predict likely languages using LSTM or k-nearest neighbor

```python
# LSTM neural network prediction (top 3)
result = instate.predict_language(names, model="lstm", top_k=3)
print(result["predicted_languages"].iloc[0])
# ['hindi', 'punjabi', 'urdu']

# K-nearest neighbor lookup (single best)
result = instate.predict_language(names, model="knn")
print(result["predicted_languages"].iloc[0])
# 'hindi'
```

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
lang_map = instate.get_state_languages(states_df)
print("State language mapping:")
print(lang_map[["state", "official_languages"]])
```

# Data

The underlying data for the package can be accessed at:
<https://doi.org/10.7910/DVN/ZXMVTJ>

# Evaluation

The v1.2.0 state model is a 2-layer character-level **bidirectional LSTM**
([`model_training/train_state_lstm.py`](https://github.com/appeler/instate/blob/main/model_training/train_state_lstm.py)),
trained on the rebuilt 34-state v2 data. On held-out surnames it reaches **~83% top-3
accuracy weighted by voter frequency** (the per-voter, real-use metric; ~60% top-1); ~78%
top-3 unweighted across all surnames (the long tail of rare/noisy surnames is harder).
Name-distinctive states score highest (Tamil Nadu, Maharashtra, Kerala, West Bengal ~0.9+);
small Hindi-belt states whose surnames overlap larger neighbours (Haryana, Himachal,
Chandigarh) are the hardest. This replaces the legacy 31-state GRU (85.3% top-3 on the older
31-state split). The model is bundled in the package — no download required.
The **language** model (`predict_language(model="lstm")`) was likewise rebuilt as a
char-BiLSTM ([`model_training/train_lang_lstm.py`](https://github.com/appeler/instate/blob/main/model_training/train_lang_lstm.py)),
replacing the legacy 3-head LSTM. Language labels are derived from each surname's state
footprint via Wikipedia official-languages-per-state, so language prediction is a
language-grouped view of the state signal. On held-out surnames it reaches **~91% top-3
weighted / ~80% unweighted** (vs the old model's 42.4% top-1 and the KNN's 67.9%).
Caveat: a few distinctive surnames whose bearers have dispersed widely (e.g. `nair`) can be
pulled toward the majority languages. Both neural models are bundled — no download required.

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
