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
| Karnataka | 15% | the 2018 parse stopped after Belgaum, Bijapur, Bagalkot, Yadgir and a quarter of Gulbarga: 9,007 of 46,549 parts | OCR of the remaining 37,500 Kannada PDFs (`karnataka_2017.tar.gz`, 23 GB), about 1.3M pages |
| Gujarat | 52% | all 33 districts present, but the Gujarati OCR recovered 64% of printed electors per part | re-OCR of 51,000 PDFs, about 1.8M pages |
| Jammu and Kashmir and Ladakh | 28% | the Urdu-only valley and Chenab districts, about 4 million electors | Urdu OCR of 9,700 PDFs |
| Lakshadweep | absent | the 2017 PDFs carry no extractable text. The CEO site's SIR final roll 2026 (64 image-only parts, `electoral_rolls/lakshadweep/lakshadweep_2026.py`, captcha read by Gemini) is downloaded and awaits OCR through the Assam 2026 pipeline (`MAL` edition added). The site's captcha-free SIR 2002 e-roll (`lakshadweep_sir2002.py`, 36,870 electors) is not used for training: it predates everyone who came of age after 2002. Its 19,104 Malayalam tokens are already in `malayalam.csv.gz` for romanizing the 2026 output | OCR of about 770 pages, tesseract first, Vision only if names fail |
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
