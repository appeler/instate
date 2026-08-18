# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Entries for releases predating this file are reconstructed from the upload
dates on [PyPI](https://pypi.org/project/instate/#history). What changed in
each was not recorded at the time, and inventing detail here would be worse
than saying so.

## Unreleased

* Add explicit prediction status reasons and model input-support metadata.
* Define deterministic train, validation, and untouched test membership and
  bind evaluation runs to data, checkpoint, membership, and label-order hashes.
* Restore the best validation epoch before saving and require its eligible
  training manifest before labeling any checkpoint evaluation as untouched test.
* Correct state and synthetic language target semantics in package metadata and
  model documentation.
* Include hermetic electoral-roll data-builder tests in the default test gate.

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
