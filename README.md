# instate: state and language composition estimates for Indian surnames

[![CI](https://github.com/appeler/instate/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/instate.svg)](https://pypi.org/project/instate)
[![Documentation](https://github.com/appeler/instate/actions/workflows/docs.yml/badge.svg)](https://github.com/appeler/instate/actions/workflows/docs.yml)
[![image](https://static.pepy.tech/badge/instate)](https://pepy.tech/project/instate)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/gojiberries/instate)

Instate looks up electoral-roll surname shares across 35 states and union
territories and estimates shares for unseen surnames with a calibrated
character model. It derives language compositions by mixing state shares
with Census 2011 mother-tongue shares.

The lookup contains 1,855,359 surname strings. Lookup and training retain
only surname-state cells with at least three source occurrences; totals
sum those retained cells. Most source rolls are from 2017, with Assam and
Lakshadweep from 2026. The outputs describe name patterns, not an
individual's residence, origin, or language.

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
#   dhingra    True               <NA>              0.530               0.231                  7583
#      sood    True               <NA>              0.194               0.364                 29451
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
the 35-state vocabulary.

## What the outputs mean

The state shares' denominator is the surname's retained occurrences across
included rolls. Cells with fewer than three occurrences are excluded before
normalization from both lookup and training; their counts do not contribute
to the published total. This is not a count of people in the current
population. The model is trained so its softmax targets exactly that distribution, and its
probabilities are temperature-scaled against held-out surnames, so the
lookup and the estimate are two routes to one quantity.

The language composition is defined, not observed:

    p(language | surname) = sum over states of
        p(state | surname) x census mother-tongue share of the language in the state

The mother-tongue shares come from Census of India 2011 table C-16, with
Telangana aggregated from its ten 2011 districts and languages below a 1%
share in every state pooled into `other`
([builder](https://github.com/appeler/instate/blob/main/model_training/build_state_language_shares.py), provenance and
hashes in the shipped manifest). Two caveats are part of the definition:
C-16 records mother tongue, not languages spoken, and the mixing assumes
language and surname are independent within a state, which understates
community-specific associations.

Known data weaknesses: Gujarat surnames remain noisy from OCR;
trailing-vowel spelling variants (Kannada `patila`,
Odia `dasa`) are merged into their canonical forms (`patil`, `das`).

## Abstention

A surname the package cannot support gets `abstained = True` and a reason
from the contract's shared vocabulary (`missing-name`, `no-letters`,
`unsupported-script`, `out-of-dictionary`, `insufficient-evidence`), never a
default distribution. Supported input is romanized ASCII `a` to `z`; the
model additionally requires three supported characters.

## Model and evaluation

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

## Data

Sources: [parsed electoral rolls](https://doi.org/10.7910/DVN/MUEGDT) and
[source PDFs](https://doi.org/10.7910/DVN/OG47IV).
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
| under 55 percent | Gujarat (52 percent, OCR loss), Jammu and Kashmir and Ladakh (28 percent, Ladakh and the Jammu region only; the Urdu valley rolls are unparsed) |
| 2026 roll | Assam (the 2026 final roll, all 126 constituencies; 113 percent of the 2019 electorate) |

Karnataka uses the recovered 2017 archive: 46,549 parts and 40,389,176
active records across 196 of 224 constituencies. The archive omits all 28
Bengaluru constituencies, AC150 through AC177, and 117 parts remain flagged
for reconciliation. Upnaam selects a Latin surname for 17,886,612 records
(44.3%) using household or relation evidence, or the explicit initials fallback,
and abstains on the rest.
After the shared filters, lookup and training retain 17,809,983 Karnataka
occurrences across 97,381 strings. Elector recovery and surname coverage
are different measures; these selective surname counts do not represent
the whole Karnataka electorate. Initials fallbacks retain a usable name
word without establishing that it is a family surname.

Lakshadweep uses 5,025 Latin surname selections from 57,618 active parsed 2026
entries; the shared lookup/training filters retain 3,312 occurrences across 381
strings. This selective sample has much lower surname coverage than the
complete box parse. The current selection sample has only four
Lakshadweep-bearing names. The lookup supplies direct evidence for covered
names. Regional diagnostics and a
comparison with the previous model are in
`model_training/karnataka_2017_model_diagnostic.json`. Chhattisgarh is not in
the vocabulary. Per-state sources, build commands, and what each gap would take
are in [`model_training/prep_er_data/SOURCES.md`](https://github.com/appeler/instate/blob/main/model_training/prep_er_data/SOURCES.md).

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
