# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Entries for releases predating this file are reconstructed from the upload
dates on [PyPI](https://pypi.org/project/instate/#history). What changed in
each was not recorded at the time, and inventing detail here would be worse
than saying so.

## 3.0.0 - Unreleased

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
