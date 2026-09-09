# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Entries for releases predating this file are reconstructed from the upload
dates on [PyPI](https://pypi.org/project/instate/#history). What changed in
each was not recorded at the time, and inventing detail here would be worse
than saying so.

## [Unreleased]

## 3.2.0 - 2026-09-08

* Replace Karnataka's partial source table with the recovered 2017 archive:
  40,389,176 active records across 46,549 parts and 196 constituencies.
  All 28 Bengaluru constituencies are absent from the source archive;
  117 parts retain reconciliation flags.
* Rebuild Karnataka surname evidence with upnaam's Kannada adapter.
  Resolve words individually through Indicate so a missing spelling elsewhere
  in a name does not discard usable surname evidence. Exclude initial-like
  tokens and preserve abstentions for single names and conflicting evidence.
  Surname confidence remains null because evidence agreement is not calibrated
  surname accuracy.
* Add 47,520 Kannada spelling pairs and repair 629 unusable entries in the
  separate Indicate corpus using structurally screened Muse Spark 1.3
  Contributor responses. Preserve native fields and keep model responses,
  screening decisions, and usage estimates in Indicate.
* Rebuild the national lookup and training targets with the same
  three-occurrence minimum, then retrain and calibrate the 35-state model
  using disjoint selection, calibration, and test surname sets.
* Report the model tradeoff on common test data: Karnataka top-one record
  coverage increases from 37.6% to 44.3%, while national top-three mass
  coverage falls from 75.0% to 66.7% and log loss rises from 1.728 to 1.820.
* Record the actual training arguments and resolved device in the training
  manifest.
* Require Python 3.12 or newer, matching the shared package baseline.

## 3.1.0 - 2026-09-07

Data release: four states rebuilt from better sources, Lakshadweep added,
and the same composition API with 35 state-share columns.

* Telangana: replace the Telugu OCR (58 percent of printed electors per
  part) with the English 2017 rolls, text PDFs parsed by
  `parse_searchable_rolls/scripts/telangana_english/parse.py` (full elector
  schema, per-part checks against the cover page: 31,128 of 31,145 parts
  match the printed ending serial, 24,592,470 active electors against
  24,592,455 printed). Surnames are resolved by household evidence, the
  token shared with a co-resident at the same house number or with the
  relation name, since these rolls mix surname-first and surname-last
  names within one part; see `name_tables.py lastnames-households` and the
  upnaam Telangana adapter; coverage 46 to
  82 percent of the 2019 electorate.
* Assam: replace the 2018 parse (40 of 126 constituencies never downloaded)
  with the 2026 final roll from `assam_elex_rolls_2026`; coverage 33 to 113
  percent.
* Jammu and Kashmir and Ladakh: add the Jammu-region Hindi roll (with repair
  of the PDF text-extraction damage to Devanagari) to the Ladakh English
  roll; coverage 3 to 28 percent. The Urdu valley rolls remain unparsed.
* Daman and Diu: the 3.0 table double-counted 78,554 electors by
  concatenating the 2015 and 2017 rolls; rebuilt from 2017 only.
* Lakshadweep: add the complete 64-part Malayalam 2026 box parse. Of
  57,618 active parsed entries (57,607 printed), upnaam selects 5,025 Latin
  surnames using household, relation, and house-name evidence. Lookup and training
  retain 3,312 occurrences across 381 strings. Original Malayalam, abstentions, reconciliation checks,
  and PDF references remain in the separate parsed-roll deposit.
* Apply the same three-occurrence minimum to surname-state cells in lookup
  and training. Normalize over retained cells only: the 1,848,011 lookup
  rows now match the training targets, and published totals exclude
  suppressed counts.
* Reserve 165,007 validation names for calibration, excluding the 20,000
  names used to select the checkpoint.
* Add Lakshadweep to the Census 2011 mother-tongue mixing matrix.
* Retrain and recalibrate the 35-state checkpoint with hash-bound training
  and test manifests. On 185,232 held-out surname inputs, modal top-1/top-3
  is 0.508/0.764, record-mass coverage is 0.469/0.751, and calibrated log
  loss is 1.724. These are metrics for the rebuilt sources; differences from
  earlier source/split memberships are not like-for-like improvements.
* Document low Lakshadweep surname coverage and model recall: the model
  does not rank Lakshadweep in its top three for any of its 38 relevant
  held-out surname inputs (350 local record weight). The lookup provides
  direct evidence for covered names.
* Document per-state coverage (README, model card,
  `model_training/prep_er_data/SOURCES.md`).
* `name_tables.py`: parquet and filtered sources, coalesced relation
  columns, a `devanagari-pdf` path, `merge`, parquet output from `ln-prop`,
  and indicate's current transliteration API.

## 3.0.0 - 2026-08-19

Breaking release: the public API is replaced. There are no
backward-compatibility aliases.

* Replace `get_state_distribution`, `predict_state`, and `predict_language`
  with three composition-form functions under appeler inference contract
  1.1: `lookup_state_composition`, `estimate_state_composition`, and
  `estimate_language_composition`. Results carry 0 to 1 shares that sum to
  one, boolean `scored`/`abstained` columns, shared abstention reasons, and
  provenance columns; unknown surnames abstain instead of returning NaN.
* Expose calibrated probabilities: the state model is temperature-scaled
  (T = 1.207) against held-out empirical state distributions, and the
  shipped checkpoint is retrained under the evaluation contract with
  published untouched-test metrics (modal top-1 0.534, top-3 0.770;
  record-weighted log loss 1.762).
* Replace the geometric ranked-official-language weights with Census of
  India 2011 C-16 mother-tongue shares per state (Telangana aggregated from
  its ten 2011 districts; languages under a 1% share in every state pooled
  into `other`), built reproducibly from hash-pinned census downloads.
* Define the language estimate as the state composition mixed with the
  census shares, replacing the separately trained language LSTM and the
  Levenshtein KNN path; drop the `Levenshtein` dependency and the 8 MB KNN
  table from the wheel.
* Rename `get_state_languages` to `lookup_state_official_languages` and
  `list_available_states` to `list_supported_states`; remove
  `get_model_metadata` and the Streamlit app.
* Earlier unreleased work: explicit prediction status reasons, deterministic
  train/validation/untouched-test membership with hash-bound evaluation
  manifests, best-validation-epoch checkpointing, corrected target
  semantics, and hermetic data-builder tests.

## 2.1.0 - 2026-08-17

* Publish neural checkpoints at immutable Hugging Face revisions instead of
  bundling them in the wheel.
* Store runtime lookup tables as typed Parquet and validate their schemas.
* Build releases with the current uv build backend and Hugging Face Hub API.

## 2.0.0 - 2026-08-15

* Require explicit name and state columns for DataFrame inputs.
* Preserve DataFrame indices and replace stale lookup columns without merge suffixes.
* Map the two pre-union electoral territory names through the shared language alias.
* Report modal-label accuracy and distribution-mass coverage from the training programs.
* Add checkpoint evaluation and repair model-building paths for the source layout.
* Count only model-supported characters toward the minimum prediction length.
* Preserve duplicates, short names, missing values, and unmatched names in
  electoral-roll lookups.
* Exclude metadata columns from the public state list.
* Validate prediction counts and retire unused download and GRU code paths.
* Load all runtime data and models from the installed package.
* Repair and test the Streamlit interface.
* Adopt the py-canon package, CI, documentation, and release structure.

## 1.2.0 - 2026-06-18

## 1.1.0 - 2025-12-27

## 1.0.0 - 2025-12-04

## 0.1.7 - 2024-08-19

## 0.1.6 - 2024-08-18

## 0.1.5 - 2024-08-18

## 0.1.4 - 2024-08-18

## 0.1.3 - 2024-08-18

## 0.1.2 - 2023-03-24

## 0.1.1 - 2023-03-15

## 0.1.0 - 2023-03-15
