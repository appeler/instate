# Per-state roll sources for the v2 name tables

Every `data/names_<slug>.csv.gz` comes from one roll file and one `name_tables.py`
subcommand. The parsed rolls live in the Parsed Indian Electoral Rolls dataset
(doi:10.7910/DVN/MUEGDT, Dataverse file ids below), the PDFs in the PDF corpus
(doi:10.7910/DVN/OG47IV). Coverage is the table's voter weight divided by the state's
2019 general-election electorate; the 2017 rolls sit at 85 to 100 percent when the
parse is complete.

## Rebuilt in 3.1

| State | Source | Why it was rebuilt | Command | Coverage before, after |
| --- | --- | --- | --- | --- |
| Daman and Diu | `daman_guj_2017.tab` (6552886) | the 3.0 table concatenated the 2015 and 2017 rolls; 78,554 electors were counted twice | `corpus --state daman --roll daman_guj_2017.tab --name-col name --father-col "father's name,husband's name,mother's name"` | 167%, 83% |
| Assam | `assam_electoral_rolls_2026_enriched.parquet` from the `assam_elex_rolls_2026` repository (2026 final roll, Cloud Vision, all 126 constituencies) | 40 of 126 constituencies never downloaded in 2018 and the OCR recovered 63% per PDF | `english --roll <parquet> --lang assam --name-col name_roman --father-col relation_name_roman --where "roll_section IN ('main','addition') AND NOT transliteration_unresolved AND NOT coalesce(deleted, false)"` | 33%, 113% (2026 electorate against 2019) |
| Jammu and Kashmir and Ladakh | `jk.tab` (3148010, English, Leh and Kargil) plus `jk_hindi_2018.tar.gz` (6709033, Hindi, the Jammu region ACs 57 to 80) | English rolls exist only for Ladakh; the Hindi parse was never used | `devanagari-pdf --roll "jk_hindi/*.csv" --lang jk_hindi --corpus hindi.csv.gz` then `merge --lang jk --inputs names_jk_english.csv.gz --inputs names_jk_hindi.csv.gz` | 3%, 28% |
| Telangana | English PDFs `eng_*.pdf` from `telangana.tar.gz` in the PDF corpus (text PDFs, one per part) | the Telugu PDFs are scans; their OCR recovered 58% of printed electors per part | `parse_searchable_rolls/scripts/telangana_english/parse.py` (full elector schema to parquet, per-part checks against the cover page) then `lastnames-households --electors electors.parquet --lang telugu` | 46%, 82% |

The Assam 2026 romanization reads Assamese with Bengali vowel values (gagoi for gogoi,
bara for bora, shaikiya for saikia). The 2018 table had the same convention through the
eroll Bengali corpus, so lookups by the conventional English spelling missed before and
still miss.

The J&K Hindi CSVs came out of PDF text extraction with damaged Devanagari: 32% of names
contain U+FFFD where a conjunct was, the i-matra precedes its consonant, and reph follows
its syllable. `repair_devanagari_pdf` undoes what is recoverable; 256,422 electors whose
surname token was damaged (mostly शर्मा and गुप्ता) are dropped. The Kashmir valley and
Chenab districts were published in Urdu only and are not parsed.

## Still short

| State | Coverage | What is missing | What it would take |
| --- | --- | --- | --- |
| Karnataka | 15% | the saved partial parse covers Belgaum, Bijapur, Bagalkot, Yadgir and part of Gulbarga (reported 9,007 parts); the download manifests list 46,549 parts for 2017 and 48,409 for 2018 | reconcile the source edition and parsed-part inventory, then process the missing districts; the legacy parser hardcodes year 2017, so its year column cannot establish the edition |
| Gujarat | 52% | all 33 districts present, but the Gujarati OCR recovered 64% of printed electors per part | re-OCR of 51,000 PDFs, about 1.8M pages |
| Jammu and Kashmir and Ladakh | 28% | the Urdu-only valley and Chenab districts, about 4 million electors | Urdu OCR of 9,700 PDFs |
| Chhattisgarh | absent | never scraped | a scrape |

## Unchanged from 3.0

The other 30 tables were built by the commands in the `name_tables.py` docstring from the
`*_all_clean+t13n.csv.gz` (Devanagari, Gujarati, Kannada, Odia, Bengali, Tamil, Telugu)
or `*_all.csv.gz` rolls; see `eroll_transliteration/eroll/states.py` for each state's
input file and corpus.

## Phases 2 and 3

```
name_tables.py lastnames --all
name_tables.py ln-prop --out src/instate/data/instate_unique_ln_state_prop_v2.parquet \
    --train-out model_training/data/instate_processed_v2.csv.gz
```

## Lakshadweep 2026

The SIR final roll has 58,528 boxes across all 64 parts. Every ending serial matches. Parsed active records total 57,618 against 57,607 printed; 29 part totals differ by up to six records. The deposit at `~/Documents/parsed_rolls/lakshadweep_2026/` contains the romanized roll, per-part checks, upnaam surname artifact, crop audit, and all 64 source PDFs in a verified tar.gz archive.

Surnames come from `upnaam resolve-electors --state lakshadweep`, revision `lakshadweep-elector-resolver-v2`. It uses household, relation and house-name evidence, preserves Malayalam vowel marks in household keys, and applies no position-only fallback. Of 57,618 active records, 5,026 receive a source surname; one lacks a Latin spelling. The 5,025 remaining selections yield 1,777 intermediate surname strings before the national ASCII/length/support filters. Repeated given names, generic house words and OCR errors remain possible. This is selective surname evidence, not 100% surname coverage.

The lookup and training file retain the same 381 Lakshadweep surname-state cells with 3,312 record weight after the per-cell minimum of three. Lookup totals exclude discarded cells. The 2002 roll contributes only Malayalam spelling pairs; it supplies no training records.

```sh
python model_training/prep_er_data/name_tables.py lastnames-upnaam --lang lakshadweep --surnames ~/Documents/parsed_rolls/lakshadweep_2026/lakshadweep_2026_surnames.parquet --out-dir data/last_names
python model_training/prep_er_data/name_tables.py ln-prop --in-dir data/last_names --out src/instate/data/instate_unique_ln_state_prop_v2.parquet --train-out model_training/data/instate_processed_v2.csv.gz
```
