# instate: state and language composition estimates for Indian surnames

[![CI](https://github.com/appeler/instate/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/instate.svg)](https://pypi.org/project/instate)
[![Documentation](https://github.com/appeler/instate/actions/workflows/docs.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/docs.yml)
[![image](https://static.pepy.tech/badge/instate)](https://pepy.tech/project/instate)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/gojiberries/instate)

Instate reports how processed occurrences of a surname distribute across
states in the 2017 Indian electoral rolls, as calibrated 0 to 1 proportions.
A lookup covers 1.9 million surnames; a calibrated character-level model
extends the same quantity to surnames outside the table; and a language
composition mixes the state shares with Census 2011 mother-tongue shares.
The outputs describe name patterns in stated reference populations. They do
not estimate an individual's residence, origin, or language.

Results follow the appeler [inference contract](https://github.com/appeler/appellation),
composition form: every row carries proportions that sum to one, explicit
abstention with a machine-readable reason instead of a default distribution,
and provenance columns identifying the exact artifacts used.

## Installation

    pip install instate

## Usage

`lookup_state_composition` reports the electoral-roll shares for surnames in
the table and abstains on the rest:

```python
import instate

result = instate.lookup_state_composition(["dhingra", "sood", "qzxv"])
result[
    [
        "surname",
        "scored",
        "abstention_reason",
        "state_share_delhi",
        "state_share_punjab",
        "surname_record_count",
    ]
]
#   surname  scored  abstention_reason  state_share_delhi  state_share_punjab  surname_record_count
#   dhingra    True               <NA>              0.529               0.231                  7586
#      sood    True               <NA>              0.194               0.364                 29455
#      qzxv   False  out-of-dictionary               <NA>                <NA>                  <NA>
```

`estimate_state_composition` runs the temperature-scaled BiLSTM for the same
quantity, including surnames the table has never seen:

```python
result = instate.estimate_state_composition(["chintalapati"])
```

`estimate_language_composition` mixes state evidence with each state's
Census 2011 mother-tongue shares. By default it uses the lookup where the
surname is known and falls back to the model, recording which in a
`language_basis` column:

```python
result = instate.estimate_language_composition(["sood", "chintalapati"])
result[["surname", "language_basis", "language_share_punjabi", "language_share_telugu"]]
```

DataFrame input uses the fleet signature: `data` first, then the column
name, with every option keyword-only.

```python
import pandas as pd

frame = pd.DataFrame({"lastname": ["sharma", "patel"], "person_id": [1, 2]})
result = instate.lookup_state_composition(frame, "lastname")
```

Two reference lookups round out the API: `lookup_state_official_languages`
maps states to their official languages, and `list_supported_states` returns
the 34-state vocabulary.

## What the outputs mean

The state shares' denominator is included, processed occurrences of the
surname in the 2017 rolls, not people in the current population. The model
is trained so its softmax targets exactly that distribution, and its
probabilities are temperature-scaled against held-out surnames, so the
lookup and the estimate are two routes to one quantity.

The language composition is defined, not observed:

    p(language | surname) = sum over states of
        p(state | surname) x census mother-tongue share of the language in the state

The mother-tongue shares come from Census of India 2011 table C-16, with
Telangana aggregated from its ten 2011 districts and languages below a 1%
share in every state pooled into `other`
([builder](model_training/build_state_language_shares.py), provenance and
hashes in the shipped manifest). Two caveats are part of the definition:
C-16 records mother tongue, not languages spoken, and the mixing assumes
language and surname are independent within a state, which understates
community-specific associations.

Known data weaknesses: Telugu/Telangana and Gujarat surnames are noisier in
the source romanization; trailing-vowel spelling variants (Kannada `patila`,
Odia `dasa`) are merged into their canonical forms (`patil`, `das`).

## Abstention

A surname the package cannot support gets `abstained = True` and a reason
from the contract's shared vocabulary (`missing-name`, `no-letters`,
`unsupported-script`, `out-of-dictionary`, `insufficient-evidence`), never a
default distribution. Supported input is romanized ASCII `a` to `z`; the
model additionally requires three supported characters.

## Model and evaluation

The state model is a two-layer character-level bidirectional LSTM trained on
the rebuilt 34-state data, with surnames assigned to deterministic disjoint
train, validation, and test splits before training and the best validation
epoch restored before saving. Training and evaluation write manifests that
bind the data bytes, checkpoint bytes, seed, and split membership;
untouched-test evaluation refuses checkpoints without an eligible manifest
([details](model_training/evaluation_contract.py)).

Shipped-checkpoint metrics on the untouched test split, 185,206 surnames
weighted by 61.4 million records:

| metric | value |
| --- | --- |
| modal state accuracy, top 1 / top 3 | 0.506 / 0.761 |
| record mass covered, top 1 / top 3 | 0.467 / 0.681 |
| record-weighted log loss, calibrated | 1.779 |
| top-1 confidence minus mass covered | -0.016 (0.060 before calibration) |

Calibration fits one temperature on the validation split against each
surname's empirical state distribution; the shipped
`instate_state_lstm_calibration.json` records the temperature, objective,
and before/after metrics.

Checkpoints and calibration download from the pinned
[Hugging Face repository](https://huggingface.co/gojiberries/instate) on
first use and are cached. Set `INSTATE_MODEL_DIR` to a directory holding the
artifacts to run offline.

## Data

The underlying electoral-roll data: <https://doi.org/10.7910/DVN/ZXMVTJ>.
Census language shares rebuild from the pinned census downloads with
`model_training/build_state_language_shares.py`.

### Coverage by state

The training rolls do not cover every state equally. Coverage below is the
table's record weight divided by the state's electorate at the 2019 general
election; a complete 2017 roll parse sits at 85 to 100 percent. Where a state
is short, the model has fewer records to learn its surnames from, and a
surname shared with a better-covered state is pulled toward that state.

| Coverage | States |
| --- | --- |
| 85 to 100 percent | Bihar, Odisha, Jharkhand, Goa, Tripura, Manipur, Maharashtra, Meghalaya, Haryana, Chandigarh, Puducherry, Punjab, Madhya Pradesh, Arunachal Pradesh, Mizoram, Uttarakhand, Sikkim, Tamil Nadu, Rajasthan, Uttar Pradesh, West Bengal, Himachal Pradesh |
| 80 to 85 percent | Nagaland, Daman and Diu, Telangana (English 2017 rolls, rebuilt in 3.1), Kerala |
| 55 to 70 percent | Dadra and Nagar Haveli, Andhra Pradesh, Delhi, Andaman and Nicobar Islands |
| under 55 percent | Gujarat (52 percent, OCR loss), Jammu and Kashmir and Ladakh (28 percent, Ladakh and the Jammu region only; the Urdu valley rolls are unparsed), Karnataka (15 percent, five northern districts only) |
| 2026 roll | Assam (the 2026 final roll, all 126 constituencies; 113 percent of the 2019 electorate) |

Chhattisgarh and Lakshadweep are not in the vocabulary. Per-state sources,
build commands, and what each gap would take are in
[`model_training/prep_er_data/SOURCES.md`](model_training/prep_er_data/SOURCES.md).

## Authors

Atul Dhingra, Gaurav Sood, and Rajashekar Chintalapati.

## Contributor Code of Conduct

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the [Contributor Code of
Conduct](https://www.contributor-covenant.org/version/1/4/code-of-conduct/).

## License

The package is released under the [MIT
License](https://opensource.org/licenses/MIT).

## Adjacent repositories

- [appeler/naampy](https://github.com/appeler/naampy) — Infer Sociodemographic Characteristics from Names Using Indian Electoral Rolls
- [appeler/ethnicolr2](https://github.com/appeler/ethnicolr2) — Ethnicolr implementation with new models in pytorch
- [appeler/parsernaam](https://github.com/appeler/parsernaam) — AI name parsing. Predict first or last name using a DL model.
- [appeler/ethnicolor](https://github.com/appeler/ethnicolor) — Race and Ethnicity based on name using data from census, voter reg. files, etc.
- [appeler/ethnicolr](https://github.com/appeler/ethnicolr) — Predict Race and Ethnicity Based on the Sequence of Characters in a Name
