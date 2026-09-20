# Per-state roll sources for the v2 name tables

Every `data/names_<slug>.csv.gz` comes from one roll file and one `name_tables.py`
subcommand. The parsed rolls live in the Parsed Indian Electoral Rolls dataset
(doi:10.7910/DVN/MUEGDT, Dataverse file ids below), the PDFs in the PDF corpus
(doi:10.7910/DVN/OG47IV). Coverage is the table's voter weight divided by the state's
2019 general-election electorate; the 2017 rolls sit at 85 to 100 percent when the
parse is complete.

## Karnataka 2017, rebuilt in 3.2

The verified `karnataka_2017` PDF archive in `10.7910/DVN/OG47IV` contains
46,549 parts. The recovered parse has 40,389,176 active records and retains
272,887 deleted records separately. It covers 196 of 224 constituencies;
all 28 Bengaluru constituencies, AC150 through AC177, are missing from the
source archive. The 2018 download manifest has the same gap and is not used
to fill this edition.

`parse_unsearchable_rolls/scripts/karnataka/` decodes the embedded Tunga font
and reconciles each part against its printed summaries. Of the 46,549 parts,
46,432 reconcile and 117 remain flagged. One flagged PDF contains only a
base roll with no printed final total; its 633 visible records are retained
with that limitation. A source identity printed twice with conflicting names
is preserved as an unresolved identity and receives no surname selection.

`upnaam resolve-electors --state karnataka` selects 17,886,612 Latin surnames
from the 40,389,176 active records (44.3%), using household and relation
evidence, with an explicit initials fallback when only one usable word remains.
Its remaining 22,502,564 rows are abstentions. The local Kannada spelling table
in `indicate` supplies romanizations; the separate Muse Spark harvest added
structurally screened spellings before this local rebuild. Provider-reported
usage and recorded rates imply $0.31 for the new harvest calls, excluding
earlier pilots. Raw responses and screening decisions remain in Indicate.
Lookup and training retain 17,809,956 occurrences across 97,381 Karnataka
surname-state cells, each with at least three occurrences.

The old Karnataka table is replaced, not added to the recovered records.
`model_training/karnataka_2017_manifest.json` records input hashes, source
coverage, abstentions, and checks that lookup and training use the same cells.

```sh
python model_training/prep_er_data/name_tables.py lastnames-upnaam --lang karnataka --surnames ~/Documents/parsed_rolls/karnataka_2017/karnataka_2017_surnames.parquet --out-dir data/last_names
python model_training/prep_er_data/name_tables.py ln-prop --in-dir data/last_names --out src/instate/data/instate_unique_ln_state_prop_v2.parquet --train-out model_training/data/instate_processed_v2.csv.gz
```

## Rebuilt in 3.1

| State | Source | Why it was rebuilt | Command | Coverage before, after |
| --- | --- | --- | --- | --- |
| Daman and Diu | `daman_guj_2017.tab` (6552886) | the 3.0 table concatenated the 2015 and 2017 rolls; 78,554 electors were counted twice | `corpus --state daman --roll daman_guj_2017.tab --name-col name --father-col "father's name,husband's name,mother's name"` | 167%, 83% |
| Assam | `assam_electoral_rolls_2026_enriched.parquet` from the `electoral_rolls_assam_2026` repository (2026 final roll, Cloud Vision, all 126 constituencies) | 40 of 126 constituencies never downloaded in 2018 and the OCR recovered 63% per PDF | `english --roll <parquet> --lang assam --name-col name_roman --father-col relation_name_roman --where "roll_section IN ('main','addition') AND NOT transliteration_unresolved AND NOT coalesce(deleted, false)"` | 33%, 113% (2026 electorate against 2019) |
| Jammu and Kashmir and Ladakh | `jk.tab` (3148010, English, Leh and Kargil) plus `jk_hindi_2018.tar.gz` (6709033, Hindi, the Jammu region ACs 57 to 80) | English rolls exist only for Ladakh; the Hindi parse was never used | `devanagari-pdf --roll "jk_hindi/*.csv" --lang jk_hindi --corpus hindi.csv.gz` then `merge --lang jk --inputs names_jk_english.csv.gz --inputs names_jk_hindi.csv.gz` | 3%, 28% |
| Telangana | English PDFs `eng_*.pdf` from `telangana.tar.gz` in the PDF corpus (text PDFs, one per part) | the Telugu PDFs are scans; their OCR recovered 58% of printed electors per part | `parse_searchable_rolls/scripts/telangana_english/parse.py` (full elector schema to parquet, per-part checks against the cover page), then `upnaam resolve-electors --state telangana` and `lastnames-upnaam --surnames surnames.parquet --lang telugu` | 46%, 82% |

The Assam 2026 romanization reads Assamese with Bengali vowel values (gagoi for gogoi,
bara for bora, shaikiya for saikia). The 2018 table had the same convention through the
eroll Bengali corpus, so lookups by the conventional English spelling missed before and
still miss.

## Telangana handoff corrected in 3.4

The final upnaam artifact contains 24,592,470 elector rows. It selects
16,394,076 Latin surname occurrences across 398,675 strings and abstains on the
remaining rows. This replaces the older fallback table, which contained
24,586,452 occurrences across 747,473 strings and was not the output of the
documented upnaam resolver.

The corrected `last_names_telugu.csv.gz` has SHA-256
`d20469593de208cedd2460c794d84020eb64c33d5678e8c29a28b5bc231544a6`.
The common national filter retains 16,091,519 Telangana occurrences; the
training and lookup artifacts reconstruct that total exactly. The handoff
comparison is retained in `data/release_preparation/state_handoffs/comparison.json`,
and the released aggregate is recorded in `../state_handoff_audit.json`.

## Jammu and Kashmir and Ladakh, rebuilt in 3.4

The 3.4 table replaces the earlier English/Hindi aggregate with audited 2018
English, Hindi and Urdu handoffs. Candidate affidavits are not used. English
contributes 66,636 selections from historical AC047–AC050. Hindi contributes
913,449 reviewed Latin selections from AC057–AC080. The calibrated Urdu handoff
preserves 4,608,102 active assembly records, selects 970,947 corroborated native
surname occurrences and maps every one of its 4,399 selected token types.

The Urdu structural inventory covers 6,014 row-bearing PDFs after excluding 24
redundant source copies. It contains 4,690,023 rows and 4,617,281 active records.
The calibrated word transfer requires at least six independent records and calls,
a 75% winning share in both, and a positive margin. Hidden controls were
1,259/1,322 exact (95.23%). Unsupported fields remain missing and their records
abstain; printed totals are never used to manufacture rows.

Hindi and Urdu share 693,201 exact one-to-one card identities. The release counts
each link once, preferring a mapped Hindi selection when both editions select and
using Urdu when it fills the linked Hindi card. Unlinked records are preserved and
no other links are inferred. After reconciliation, J&K supplies 1,947,771 input
occurrences across 6,538 Latin strings: 66,636 English, 913,449 Hindi and 967,686
Urdu. The common national cell filter retains 1,942,682 J&K occurrences.

The shared Urdu corpus contains 27,221 native/Latin pairs and all 4,399 selected
Urdu types. Indicate rebuilds the same 27,221-key local lookup. Model-only spellings
remain silver annotations; the map and source-reading control set do not establish
surname or transliteration accuracy.

Build and validation are reproducible with:

```sh
python -m model_training.prep_er_data.build_jk_recovery_release \
  --base-inputs data/release_preparation/jk_urdu_inputs \
  --output data/release_preparation/final_recovery_inputs \
  --english data/jk_recovery/english_upnaam_verified/surnames.parquet \
  --english-audit data/jk_recovery/english_upnaam_verified/audit.json \
  --hindi data/jk_recovery/hindi_reviewed_candidate/surnames.parquet \
  --hindi-audit data/jk_recovery/hindi_reviewed_candidate/audit.json \
  --urdu data/jk_recovery/urdu_calibrated_recovery/upnaam_mapped/surnames.parquet \
  --urdu-audit data/jk_recovery/urdu_calibrated_recovery/upnaam_mapped/audit.json \
  --links data/jk_recovery/cross_script_edition_links.parquet \
  --links-audit data/jk_recovery/cross_script_card_identity.json
python -m model_training.prep_er_data.name_tables ln-prop \
  --in-dir data/release_preparation/final_recovery_inputs \
  --out data/release_preparation/final_recovery_candidate/lookup.parquet \
  --train-out data/release_preparation/final_recovery_candidate/training.csv.gz
python -m model_training.prep_er_data.validate_national_candidate \
  --inputs data/release_preparation/final_recovery_inputs \
  --lookup data/release_preparation/final_recovery_candidate/lookup.parquet \
  --training data/release_preparation/final_recovery_candidate/training.csv.gz \
  --output data/release_preparation/final_recovery_candidate/validation.json
```

The final validator reports zero duplicate, schema, support, probability,
surname-state reconstruction or state-total errors across all 35 states. Detailed
source and mapping evidence is in `../jk_2018_*_audit.json`,
`../state_handoff_audit.json`, and
[`../coverage_recovery_plan.md`](../coverage_recovery_plan.md).

## Lakshadweep 2026

The SIR final roll has 58,528 boxes across all 64 parts. Every ending serial matches. Parsed active records total 57,618 against 57,607 printed; 29 part totals differ by up to six records. The deposit at `~/Documents/parsed_rolls/lakshadweep_2026/` contains the romanized roll, per-part checks, upnaam surname artifact, crop audit, and all 64 source PDFs in a verified tar.gz archive.

Surnames come from `upnaam resolve-electors --state lakshadweep`, revision `lakshadweep-elector-resolver-v2`. It uses household, relation and house-name evidence, preserves Malayalam vowel marks in household keys, and applies no position-only fallback. Of 57,618 active records, 5,026 receive a source surname; one lacks a Latin spelling. The 5,025 remaining selections yield 1,777 intermediate surname strings before the national ASCII/length/support filters. Repeated given names, generic house words and OCR errors remain possible. This is selective surname evidence, not 100% surname coverage.

The lookup and training file retain the same 381 Lakshadweep surname-state cells with 3,311 record weight after the per-cell minimum of three. Lookup totals exclude discarded cells. The 2002 roll contributes only Malayalam spelling pairs; it supplies no training records.

```sh
python model_training/prep_er_data/name_tables.py lastnames-upnaam --lang lakshadweep --surnames ~/Documents/parsed_rolls/lakshadweep_2026/lakshadweep_2026_surnames.parquet --out-dir data/last_names
python model_training/prep_er_data/name_tables.py ln-prop --in-dir data/last_names --out src/instate/data/instate_unique_ln_state_prop_v2.parquet --train-out model_training/data/instate_processed_v2.csv.gz
```

### Karnataka initials provenance

Resolver v4 records initials-plus-word selections without corroboration as
`initials_single_token`; their confidence remains null. They are usable name
tokens, not verified hereditary surnames. The native single-word group remains
excluded. A source-image follow-up checked 26 such entries across retained
crop samples; none visibly lost a separate name word, but this is not a
population parsing-accuracy estimate.
