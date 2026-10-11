# Jammu and Kashmir electoral rolls, 2018

The full roll export follows the column names in
[`in-rolls/electoral_rolls_parse_searchable`](https://github.com/in-rolls/electoral_rolls_parse_searchable)
(`README.md` and `scripts/jk_hindi.py`, revision
`b2f692101a619d258bff2ad104a0408b7e8ddc3e`). The 27 base columns combine
the README contract with the Hindi parser’s amendment flag. It includes the territory of Jammu and
Kashmir as recorded in 2018, including Ladakh. These are electoral registration
records, not a census or a verified count of distinct people.

The completed export passed 47 independent structural and provenance checks.
Consult `jk_2018_completion.json` for the complete evidence record. The published Instate 3.6.0 surname inputs are retained
separately from the enriched full rolls.

## Files and observation units

| File | Observation unit |
| --- | --- |
| `jk_2018_{english,hindi,urdu}.parquet` | One reconciled entry in a source language edition, including inactive and non-assembly entries |
| `jk_2018_{english,hindi,urdu}_active.parquet` | One active, assembly-eligible entry in that edition |
| `jk_2018_active.parquet` | Active edition union, retaining Hindi for each accepted Hindi–Urdu link |
| `edition_links.parquet` | One accepted pair of Hindi and Urdu record IDs |
| `reconciled_active_index.parquet` | Record ID and language for each row in the active edition union |
| `urdu_enriched_inventory.parquet` | Original Urdu inventory with verified house, age, and printed sex evidence added |
| `*_parts_audit.json` | Source-part counts and differences from available printed closing totals |
| `audit.json`, `SCHEMA.json`, `SHA256SUMS` | Input provenance, coverage, physical schema, and output checksums |

Correction and deletion events have already been reconciled. Do not append
correction cards as additional electors or subtract deletion totals again.
`active` and `assembly_eligible` are separate flags: closing printed totals can
include non-permanent-resident entries that are excluded from assembly exports.
Both language versions of linked records remain in their edition files. The
edition union resolves the accepted links only; it does not establish uniqueness
of people across all sources.

## Export coverage

The source audits cover 545 English, 3,146 Hindi, and 6,014 Urdu parts; one
Hindi source part contains no entries and remains in its part audit.
The three edition files contain **7,177,547** reconciled entries. Their active,
assembly-eligible subsets contain 7,008,268 entries before reconciling 693,201
accepted Hindi–Urdu links. The combined active file contains **6,315,067** rows.

| Edition | All reconciled entries | Active assembly entries | Native elector names | Complete Latin names | House identifiers | Ages | Printed sex |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 171,653 | 169,191 | 171,612 | 171,612 | 171,644 | 168,760 | 171,653 |
| Hindi | 2,315,871 | 2,230,975 | 2,258,522 | 1,572,626 | 2,304,146 | 2,278,154 | 2,311,456 |
| Urdu | 4,690,023 | 4,608,102 | 2,315,587 | 2,147,067 | 4,225,187 | 4,169,381 | 4,144,793 |

Field counts in this table use **all reconciled entries**, including inactive
and non-assembly entries. The export audit supplies active assembly coverage by
constituency. Complete Latin-name coverage requires every token to be supported;
it is separate from the surname-selection counts used by Instate.

The typed contract is also available in `jk_2018_schema.json`. The local final
files are in `data/jk_2018_rolls_final/`; the verified external deposit is
`/Volumes/Staging/parsed_rolls/jk_2018/`. The older surname-only deposit is retained
under `/Volumes/Staging/parsed_rolls/history/jk_2018_surname_only_20261010/`.

## Basic schema

Every column is nullable unless its identity or state flag is required by the
builder. Missing values are null, not empty strings, zeros, or inferred values.
Identifiers are strings so leading zeros survive.

| Column | Type | Meaning |
| --- | --- | --- |
| `number` | string | Printed entry serial number |
| `id` | string | Printed elector ID; not assumed unique across the dataset |
| `elector_name` | string | Accepted name in the source script |
| `father_or_husband_name` | string | Accepted relative name; consult `relationship` because the label can also denote a mother or another relation |
| `husband` | boolean | True for an explicit husband relation, false for another recognized relation, null for an unresolved relation |
| `house_no` | string | Printed house identifier, preserving separators and zero values |
| `age` | int32 | Parsed printed age; anomalous values remain visible with `age_issue` |
| `sex` | string | Printed category: `Male`, `Female`, `Third Gender`, or null |
| `ac_name` | string | Assembly constituency name from the source polling-station manifest |
| `parl_constituency` | string | Parliamentary constituency; unavailable in this export |
| `part_no` | string | Source polling-station/part number |
| `year` | int32 | 2018 |
| `state` | string | `JK`, following the historical source geography |
| `filename` | string | Original PDF filename |
| `main_town`, `police_station`, `mandal`, `revenue_division` | string | Unavailable source metadata; null |
| `district` | string | District name from the source manifest |
| `pin_code` | string | Unavailable; null |
| `polling_station_name` | string | Station name from the source manifest |
| `polling_station_address` | string | Unavailable; a station name is not substituted for an address |
| `net_electors_male`, `net_electors_female` | int32 | Verified printed closing controls when unambiguous and arithmetically valid |
| `net_electors_third_gender` | int32 | Unavailable printed control; null even where individual third-gender records exist |
| `net_electors_total` | int32 | Accepted printed closing total, including non-assembly entries where present |
| `original_or_amendment` | string | `original` for the latest base card; `amendment` for a latest addition or correction |

Printed controls repeat on every record in a part. Never sum these repeated
columns across elector rows. Use one control per source filename and compare
like populations, including the treatment of non-assembly entries.

The export audit also reports active assembly coverage by constituency, missing
and repeated elector IDs, and out-of-range ages. Repeated IDs are diagnostic;
they do not authorize deleting records or treating them as the same person.

## Additional fields

`record_id` combines edition and stable `entry_event_key`. `event_key` identifies
the latest reconciled source card. `event_type`, `supplement`,
`change_event_keys`, `active`, `assembly_eligible`, and `deleted_stamp` preserve
the event history and inclusion decisions. `source_sha256`, `filename`,
`source_page` (one-based PDF page), and `source_bbox` locate the supporting card.
`language` names the edition and `assembly_constituency` is its numeric identifier.

`elector_name_native` and `relative_name_native` preserve accepted source text.
`elector_name_roman` and `relative_name_roman` contain complete local Indicate
lookup renderings or null. Their corresponding `_roman_status` columns distinguish
`source_latin`, `lookup_complete`, `missing_native`, and `unresolved_token`.
Unknown tokens do not produce partially transliterated names. Previously reviewed
Hindi exclusions override generic lookup matches. A lookup rendering is not an
independently verified preferred spelling of an individual's name.

`relationship` is a recognized relation category; `relationship_label_native`
preserves the printed label. `name_issue` and `relative_issue` preserve upstream
extraction issues. `age_raw` and `sex_raw` retain evidence before normalization.
`age_issue` distinguishes missing, unparsed, and values outside 18–120 without
silently deleting anomalous printed ages. `house_source`, `age_source`, and
`sex_source` identify the evidence route. `demographic_issue` records a failure
in the additional Urdu outline pass; a separate verified dot-font overlay may
still supply a field on that record.

## Verification contract

The builder checks inventory hashes against source audits, source-card hashes,
polling-station filename identity, unique entry keys, exact demographic-overlay
identity, conflicting demographic values, part-level row and activity counts,
and one-to-one active edition links. Demographic joins include source hash,
filename, stable and latest event keys, serial, elector ID, page, and activity
flags. A duplicate key, unmatched overlay, or conflicting non-null value stops
the build.

Urdu demographic recovery requires recognized printed field labels before
assigning values. Numeric glyphs require supported physical outlines or an
explicit supported unembedded WinAnsi encoding. Ambiguous fields, unreadable
fonts, and image overlaps cause abstention. Malformed embedded fonts preserve
all entry identities while leaving the affected page's new demographics null.
An unresolved field is not proof that the printed source lacks that field.

Source coverage gaps and pre-existing printed-total discrepancies remain in the
audits: 151 English, 58 Hindi, and 340 Urdu parts disagree with available
printed closing totals. These differences are retained rather than repaired by
adding or deleting registrations. Coverage, successful joins, and passing arithmetic checks do not establish
extraction accuracy. Independent source-card checks are retained separately from
development examples and automated structural tests.

The frozen fresh Urdu holdout contains 30 cards from 20 parts and seven stacked
layouts. All 74 emitted fields matched the source in independent visual checks
(25 house identifiers, 22 ages, 27 sex labels). Eleven readable fields remained
withheld. Nineteen of 20 active totals matched closing controls; the remaining
+33 discrepancy preceded enrichment. These deliberately stratified checks are
not a corpus-wide accuracy estimate. In one part, all 339 printed male entries
were recovered but only 127 of 306 female entries were recovered. Missingness
must therefore be assessed before demographic analysis, and cannot be treated
as random. See `jk_2018_urdu_demographic_validation.json`.

## Rebuild

With the retained private inputs in `data/jk_2018`, the local Indicate checkout,
and Instate's roll-recovery dependencies installed, rebuild into a new output
directory. This ordinary rebuild reuses the retained demographic overlay:

```sh
PYTHONPATH=.:../indicate .venv/bin/python -m model_training.prep_er_data.build_jk_rolls \
  --root data/jk_2018 --out data/jk_2018_rolls_rebuilt
cd data/jk_2018_rolls_rebuilt
shasum -a 256 -c SHA256SUMS
```

The output directory must be new. Each DuckDB connection uses a separate
temporary spill directory so simultaneous jobs cannot overwrite a live reader.
The source inventories, elector-level outputs,
review crops, and transliteration corpora remain private and ignored by Git.
Public repository artifacts contain code, schemas, aggregate audits, and hashes.

To regenerate the additional Urdu overlay from archived PDFs, first prepare a
fresh copy of `data/jk_2018` at `data/jk_2018_regeneration_inputs`, excluding
`urdu/demographics_outline.parquet` and `urdu/demographics_outline_audit.json`.
Keep the retained input directory intact. From the repository root, run the
frozen extractor, then pass its completed directory to the builder. Both output
directories below must be new:

```sh
PYTHONPATH=. .venv/bin/python -m model_training.prep_er_data.jk_urdu_demographics \
  --inventory data/jk_2018_regeneration_inputs/urdu/inventory.parquet \
  --pdf-root /Volumes/Staging/instate/jk_recovery_archive_20261010/raw/urdu_full \
  --reference-font '/Volumes/Staging/instate/jk_recovery_archive_20261010/reference_fonts/Nafees Nastaleeq(Updated).ttf' \
  --colon-catalog /Volumes/Staging/instate/jk_recovery_archive_20261010/muse_review/urdu_ocr_pilot/glyph_verification.json \
  --out data/jk_2018_regenerated_demographics --workers 6
PYTHONPATH=.:../indicate .venv/bin/python -m model_training.prep_er_data.build_jk_rolls \
  --root data/jk_2018_regeneration_inputs --out data/jk_2018_regenerated_rolls \
  --demographic-recovery data/jk_2018_regenerated_demographics
```

The extractor supports `--resume` only with identical input, code, and dependency
hashes. Consolidation verifies every part checkpoint and requires exactly the
canonical inventory's entry-key universe. It writes
`urdu/demographics_outline.parquet` and its audit once; subsequent roll builds
reuse those files without `--demographic-recovery`. The earlier verified
dot-font evidence remains in `urdu/demographics.parquet`; values from the two
passes are combined only if all non-null evidence agrees.

## Enriched surname handoff and national candidate

The demographic export feeds Upnaam's `jk-urdu-2018-demographics` source profile
and resolver v5. The calibrated v4 handoff remains the published 3.6 baseline.
Run the enriched handoff from the Upnaam repository:

```sh
.venv/bin/upnaam resolve-jk-urdu \
  ../instate/data/jk_2018_rolls_final/urdu_enriched_inventory.parquet \
  ../instate/data/jk_2018_rolls_final/urdu_enriched_audit.json \
  ../instate/data/jk_2018/urdu/enriched_handoff \
  --romanization-map ../instate/data/jk_2018/urdu/romanization_map.json
```

Then run from Instate, choosing new output directories on a repeat build:

```sh
PYTHONPATH=. .venv/bin/python -m model_training.prep_er_data.build_jk_recovery_release \
  --base-inputs data/release_preparation/final_recovery_inputs \
  --output data/release_preparation/jk_transliteration_inputs \
  --english data/jk_2018/english/handoff/surnames.parquet \
  --english-audit data/jk_2018/english/handoff/audit.json \
  --hindi data/jk_2018/hindi/handoff/surnames.parquet \
  --hindi-audit data/jk_2018/hindi/handoff/audit.json \
  --urdu data/jk_2018/urdu/enriched_handoff/surnames.parquet \
  --urdu-audit data/jk_2018/urdu/enriched_handoff/audit.json \
  --links data/jk_2018/edition_links.parquet \
  --links-audit data/jk_2018/edition_links_audit.json
PYTHONPATH=. .venv/bin/python -m model_training.prep_er_data.name_tables ln-prop \
  --in-dir data/release_preparation/jk_transliteration_inputs \
  --out data/release_preparation/jk_transliteration_candidate/lookup.parquet \
  --train-out data/release_preparation/jk_transliteration_candidate/training.csv.gz
PYTHONPATH=. .venv/bin/python -m model_training.prep_er_data.validate_national_candidate \
  --inputs data/release_preparation/jk_transliteration_inputs \
  --lookup data/release_preparation/jk_transliteration_candidate/lookup.parquet \
  --training data/release_preparation/jk_transliteration_candidate/training.csv.gz \
  --output data/release_preparation/jk_transliteration_candidate/validation.json
```

Household corroboration can change selections or introduce conflicting evidence.
A review of newly selected tokens found 218 selections of designation words in
the initial candidate. The corrected profile excludes ten exact forms supported
by dictionary definitions and source-card review, records the exclusions in its audit, and preserves raw names.
See `jk_2018_title_review.json`; the development output is retained separately.
The comparison records gains, losses, and changed native selections against the
published baseline. Additional corroboration is not independently verified
hereditary-surname accuracy. Confidence remains null. The new national candidate
is unreleased; the published lookup, neural model, and calibration retain their
existing revisions.

The corrected Urdu handoff retains all 4,608,102 active assembly entries and
939,660 source-defined households. It selects 1,371,913 native tokens, maps
1,371,162 of them, and preserves 751 unmapped selections. It abstains on
3,236,189 records, including 34,972 conflicts between evidence sources.
Against the published calibrated baseline, it gains 435,944 native selections,
loses 34,978, and changes 18,008 retained selections. These changes combine new
household evidence and the documented designation exclusions; they are not
an accuracy improvement estimate. See `jk_2018_enriched_comparison.json` and
`jk_2018_enriched_urdu_audit.json`.

The resulting national candidate passes all eleven validation checks across
35 states. Its 1,762,393 surname strings and 2,596,211 surname-state cells retain
737,261,134 record occurrences. J&K supplies 2,347,226 reconciled mapped
selections; 2,341,907 remain after the shared national filters. The candidate
lookup SHA-256 is
`fdb1cf144f6a924552b0d0f2442717d693f9ef8961667c6a07204567efda4afd`.
The validation record is `jk_2018_national_candidate_validation.json`.

## Urdu transliteration extension

A three-pass Muse Spark Contributor review covered 179 unmapped native token
types selected by the enriched handoff (1,637 occurrences). Six requests passed
all 48 positive and 24 negative controls. The strict selection policy added 61
mappings covering 886 selected occurrences, while preserving every existing
mapping. Thirty-one additions have lexical support; 30 are model-only.
Nine otherwise eligible types with historical source warnings were withheld.
The remaining 118 types cover 751 unmapped selections.

Each new mapping requires three distinct supported, unflagged passes, a repeated
ASCII primary spelling, agreement with attested lexical forms where available,
and no historical source warning. Same-model agreement is correlated evidence,
not measured pronunciation accuracy. The lexical reference uses all Dakshina
splits and is not a held-out evaluation. The preserved calibrated baseline
contains 589 mapped decisions carrying source warnings; the stricter rule applies
to this extension only.

The expanded map contains 27,282 pairs. The retained decision table has 45,444
rows, while the review ledger covers 45,449 types: five newly reviewed unresolved
types remain only in the ledger. The extension produces 126,457 additional
complete Latin elector names and 12,815 additional complete relative names.
Across all Urdu edition rows, 168,520 accepted elector names and 164,752 accepted
relative names still lack a complete Latin rendering. Native text, demographic
fields, selection evidence, source identities and row inclusion do not change.

Provider-reported usage implies an estimated $0.007890396 for this extension,
bringing cumulative accounted exposure to $9.20387145 against the authorized
$10 cap. Only standalone tokens and controls were submitted. Local Indicate
lookups consume the reviewed corpus without paid calls. The policy, hashes,
controls and downstream checks are recorded in
`jk_2018_urdu_transliteration_extension.json`.

## Archive and local cleanup

The original `data/jk_recovery/` tree was removed only after all 18,788 files
matched the archived source by SHA-256. The source archive remains at
`/Volumes/Staging/instate/jk_recovery_archive_20261010/`.
The completed snapshot at
`/Volumes/Staging/instate/jk_2018_completed_20261010/` contains the compact
inputs, national candidate, full extraction checkpoints, development runs, and
independent verification code, and producing source-code archives. Its manifest records 12,537 files and their hashes.
The old local full-roll prototype and demographic checkpoints/pilots were also
removed after comparison with their archived copies. The retained local bundle
and final rolls pass `make verify-jk`; neither depends on the deleted working tree.
Cleanup increased available disk space by 18,513,920,000 bytes (about 17.24 GiB).

The transliteration extension is archived separately at
`/Volumes/Staging/instate/jk_2018_transliteration_20261010/`. Its `build_inputs/`
directory preserves the exact roll-build inputs; `retained_inputs/` includes the
review ledger and downstream handoff. The older completed snapshot remains
immutable. The immediately preceding full-roll deposit is retained under
`/Volumes/Staging/parsed_rolls/history/jk_2018_before_transliteration_20261010/`.

The enriched handoff was generated from the preceding export's inventory.
Independent keyed comparisons verify unchanged inventory rows in the rebuilt
export. Its exact source inventory, audit and producing builder are retained in
`data/jk_2018/urdu/enriched_transliteration_review_20261010/` as
`handoff_source_inventory.parquet`, `handoff_source_audit.json` and
`handoff_source_builder.py`. The rebuilt export
records the later spill-isolation fix in its own source hashes; this changes
provenance without changing the inventory or native evidence.
