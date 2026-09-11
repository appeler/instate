# Surname evidence and state coverage recovery

Relative-name evidence and isolated coverage adjustment are implemented. Current
parser execution order: Jammu and Kashmir and Ladakh, following the Andaman and
Dadra rebuilds included in 3.3.0. The complete historical archive is verified,
and all 3,145 readable Hindi PDFs are rebuilt and audited. Their absolute
discrepancy against available printed totals falls from 69,320 to 145.
The original English archive is also rebuilt, including its supplements;
its remaining source-count discrepancies are documented separately below.

## Next release preparation, September 11

J&K English and Hindi now have per-elector selections and abstentions in upnaam.
English has a Latin count table; Hindi still needs validated transliteration.
Urdu names are unavailable. None of these new J&K artifacts is in the runtime lookup.

The state handoff audit found that Lakshadweep's 1,777 saved surname strings and
5,025 selected occurrences match the upnaam artifact exactly. Telangana's saved
table does not: it contains 24,586,452 occurrences from the old fallback resolver,
while its upnaam artifact selects 16,394,076. See `state_handoff_audit.json`.
The two artifacts describe the same 24,592,470 active electoral records.

A corrected Telangana candidate is staged under
`data/release_preparation/telangana_candidate/`. The unchanged 35 input tables
reproduce every row of the 3.3.0 lookup. Replacing only the Telangana input yields
1,825,288 surname strings, 2,725,718 retained cells and 730,097,169 occurrences.
Telangana retains 16,091,518 occurrences after filters, compared with 23,977,891
in 3.3.0. The existing national spelling reconciliation changes when Telangana
counts change; all other states together lose a further 730 retained occurrences.
Lookup and training counts agree exactly, shares sum to one, and all retained
cells meet the three-occurrence floor. This candidate is not published.

The duplicate household resolver and `lastnames-households` command are removed
from instate. Elector resolution belongs in upnaam; instate counts its artifacts.
The legacy position-based builder remains necessary to reproduce older state
sources. Parquet requests now write only the requested Parquet file, without a
second CSV lookup, and input/output paths containing apostrophes work. PDF recovery
dependencies are declared in the `roll-recovery` group and included in tests.

Superseded development reports and the 3.1 Lakshadweep model diagnostics are in
`history/`. The current parser, source audits, frozen full-corpus splits and final
handoff manifests remain active. Source PDFs, person-level rows and local model
artifacts remain outside Git.

The authorized release scope is to complete the J&K artifacts and publish the next
instate release. Muse Spark Contributor access for independent review is still
pending. No release tag, runtime-artifact promotion or publication has occurred.

The Urdu decoder now reconstructs font-defined letter bodies and dots, retaining
all candidates that reproduce the exact source outlines. On physical page 3 of
the frozen pilot, it finds unique text for 106 of 1,723 development span segments
and 1,478 of 34,255 validation segments. These are PDF span fragments, not complete
names. Unmatched outlines remain the main obstacle; approximate shape matches are
diagnostic only. No Urdu name has been promoted into an elector artifact.
The decoder also passes all 1,806 synthetic single-letter and two-letter
roundtrips: 1,668 are unique and 138 retain multiple indistinguishable spellings.

Hindi transliteration review inputs are staged locally: 5,857 distinct native
selections and a frozen 152-token diagnostic pilot. Existing corpora agree on
2,838 strings representing 1,134,578 selected occurrences; 1,307 strings occur in
only one corpus, 1,679 are missing, 31 have invalid Latin candidates, and two have
conflicting candidates. These are candidate coverage measures, not validation.
No model call or accepted romanization has been produced by this review stage.

A record-level comparison of the seven bilingual pilot parts adds evidence beyond
their matching closing totals. Three match every appearance on page, serial,
identifier and deletion stamp. Three more have identical appearance multisets
when page numbers are omitted, including two with missing identifiers. The
remaining Urdu PDF lacks 30 appearances, consistent with its documented page gap.
No cross-language field transfer or deduplication is applied. Details are in
`jk_2018_urdu_hindi_controls.json`; decoder results are in
`jk_2018_urdu_font_audit.json`. Local checks pass: 243 tests, one live check
deselected, 98.11% runtime coverage, formatting, linting, typing and docstrings.

## J&K recovery checkpoint, September 10

The 3.3.0 release is complete. Earlier experiments and release-preparation
notes are preserved in [history](history/coverage_recovery.md). J&K recovery is now in progress. Source counts and hashes are
in `jk_2018_source_audit.json`.

The full raw archive has 14,051 PDF-named files: 3,146 Hindi-prefixed,
545 English-prefixed and 10,360 Urdu-prefixed files. Only 9,733 open as PDFs:
one Hindi file is empty, and 4,317 Urdu files contain only zero bytes.
The 10,402 distinct AC/part filename keys include 1,547 with no readable PDF
and 878 with readable Hindi and Urdu versions. These are archive inventory
counts, not a complete official part catalogue or an electorate coverage rate.
Language versions must be compared before combining. The source checksum,
missing keys, overlaps and constituency counts are in `jk_2018_archive_audit.json`.

The cleaned English archive has 540 English PDFs and two Urdu PDFs. Dataverse files
6898963 and 6894542 are identical archives, not independent sources. The
English archive, original parsed CSV (3148010), and Hindi CSV archive (6709033)
were verified against their published MD5 checksums. The Dataverse-generated
TSV differs from the original CSV; the original download is the verified baseline.

The repaired reader in `../parse_searchable_rolls` addresses three problems:

- Relationship labels `Self`, `Son in Law`, and `Adopted Son` caused 295 named
  boxes to be dropped. Their literal types now survive extraction; they are
  not reclassified as father or husband evidence.
- Another 54 numbered boxes have no extracted name. They remain in the artifact
  with `source_name_missing=true` and cannot supply a recorded surname.
- The 22 NPR boxes use a different layout. Their house numbers were read as
  elector serials, overwriting 22 other records. Their printed serials, names,
  relative names, ages and houses now parse separately. Both layouts are
  exported, with `assembly_eligible=false` for NPR records. That flag is not a
  deletion flag or a statement about parliamentary eligibility.

| English source measure | Original parse | Repaired parse |
| --- | ---: | ---: |
| Retained records | 164,713 | 165,084 |
| Main-roll records | Not separately reliable | 165,062 |
| NPR records | Misparsed and overwriting other rows | 22 |
| Blank-name records retained | 0 | 54 |

All 540 English PDFs were rebuilt. The two Urdu PDFs remain reader errors.
Of the English parts, 390 match the extracted printed totals and 150 remain
discrepant. Printed totals sum to 163,986; net and absolute discrepancies are
both 1,098 when both layouts are counted. Visible boxes and printed summaries
do not always agree; the remaining differences have not all been attributed.
Do not infer deletions or choose a denominator solely to remove those differences.

The full raw archive contains five additional English parts. All 540 shared PDFs
differ from the cleaned archive and contain more pages: 11,795 versus 8,316 pages.
The five additional parts bring the raw English total to 11,883 pages. Most raw
originals use embedded CFF fonts with damaged character mappings; eight use
TrueType subsets. The cleaned-archive reader does not decode these originals.
The September 11 rebuild below uses reference font outlines and reconstructs
the full ledger. The earlier glyph-alignment probe was not promoted into a
character map: the cleaned and original PDFs contain different source versions.

```sh
PYTHONDONTWRITEBYTECODE=1 python model_training/prep_er_data/rebuild_jk_english.py \
  --pdf-dir data/jk_recovery/raw/english \
  --parser-root ../parse_searchable_rolls \
  --out-dir data/jk_recovery/rebuilt
```

The output directory must be new. The Parquet preserves raw box text, source
numbers, identifiers, relationship labels, missing-name status and eligibility.
`record_key` combines the PDF filename with a row ordinal; it does not assume
source serials are unique. `audit.json` records the schema, PDF hashes, reader
failures, per-part discrepancies, parser hashes and artifact hash. Source fields
remain strings, including ages and houses. Outputs are restricted local
artifacts under the git-ignored `data/` directory. The parser CSV schema now
includes `assembly_eligible`; other state readers leave it blank. Existing CSVs
with an older header cannot be appended to.

### Original English ledger, September 11

`english_glyphs.py` matches embedded CFF and TrueType outlines against the ASCII
cmaps of local Arial and Times New Roman reference fonts. It requires agreement
in advance width, bounds and both directions of the rasterized contours, within
four pixels at 1,000 pixels/em. Subset glyph numbers and names do not establish
character identity. Missing, ambiguous and `.notdef` glyphs remain unavailable.
This rendering check is approximate and is not a name-accuracy estimate. The
implementation follows the fontTools [pen protocol](https://fonttools.readthedocs.io/en/latest/pens/basePen.html)
and [CFF reader](https://fonttools.readthedocs.io/en/latest/cffLib/index.html).
Reference binaries remain local; their SHA-256 hashes are in the audit.

`jk_english_rows.py` retains source boxes, raw text, decoded candidates, literal
relationships and eligibility. Wrapped elector and relative names stay together.
Addition, deletion and correction headings control the ledger; a correction
replaces fields without adding an elector, and a deletion stamp plus a deletion
list entry subtracts once. Numbered blank-name records remain countable. NPR
serials use a separate namespace and cannot overwrite assembly records.

All 545 PDFs now produce events. The 176,992 source appearances comprise
165,996 base records, 5,658 additions, 2,440 deletions and 2,898 corrections.
They yield 171,653 inventory identities and 169,213 active records, including
22 NPR records. Accepted names are available for 169,174 active records; the
remaining 39 have no name. The TrueType repair recovers eight previously unread
parts and leaves all event and inventory rows in the other 537 parts identical.

| Original English comparison | Result |
| --- | ---: |
| Parts with a closing total | 543 |
| Parts matching that final total | 392 |
| Printed final electorate in comparable parts | 167,115 |
| Recovered active records in those parts | 168,193 |
| Net difference | +1,078 |
| Sum of absolute per-part differences | 1,080 |

These are comparisons within the original PDFs. They cannot be interpreted as
an improvement rate against the cleaned archive's 540-part totals. For example,
EACA047PS0001's cleaned closing page is dated October 3, 2017 and reports 89;
the original's January 19, 2018 closing page reports 86 after three deletions.

EACA048PS0053 contains only 14 of its printed 20 pages, and EACA049PS0077 contains
33 of 38. Their 1,020 active records remain in the inventory but outside the
closing-total comparison. EACA048PS0077 repeats an addition with the same serial,
ID, name and house on page 35; the event ledger retains both appearances and the
inventory counts the identity once, leaving a final discrepancy of minus one.
EACA048PS0217's correction page visibly contains no elector and prints zero
modifications, while its closing table reports one. Component discrepancies
remain explicit; matching a final total does not validate all name updates.

The output is `data/jk_recovery/english_full_verified/`; the compact checkpoint
is `jk_2018_english_full_audit.json`. The audit records source and artifact hashes,
reference-font hashes, per-page extraction counts, source arithmetic, component counts,
page gaps and unresolved issues. A conditional crop review covers 16 accepted
names and four missing-name records; all agree with the visible source. This
small single-reviewer check is not a population accuracy estimate. The source
itself identifies its Urdu version as authoritative.

```sh
uv sync --group roll-recovery
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m model_training.prep_er_data.jk_english_rows \
  --pdf-dir data/jk_recovery/raw/english_full \
  --reference-font '/System/Library/Fonts/Supplemental/Arial.ttf' \
  --reference-font '/System/Library/Fonts/Supplemental/Arial Bold.ttf' \
  --reference-font '/System/Library/Fonts/Supplemental/Times New Roman.ttf' \
  --reference-font '/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf' \
  --reference-font '/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf' \
  --split model_training/jk_2018_english_full_split.json \
  --out-dir data/jk_recovery/english_full_verified \
  --workers 4
```

Use a new output directory. Source PDFs, elector rows and crop reviews remain
local. The English parts cover historical J&K AC047–AC050, now Ladakh; do not
combine them with a modern J&K denominator without defining the geography.
The English upnaam handoff below validates source rows and preserves abstentions.
Urdu extraction, missing sources and surname-accuracy evaluation remain work.

### Urdu controls and extraction pilot, September 11

The closing-page pass inspected all 6,043 readable Urdu sources with isolated
per-PDF processes. It checks the last two pages because a closing table can
precede its dated signature. It accepts the observed eight-row table with
readable Latin numbers; anonymous native-font text never supplies names.
Source and artifact hashes, independent arithmetic checks and per-part results
are in `jk_2018_urdu_full_closing_audit.json`.

| Urdu source-control result | Count |
| --- | ---: |
| Parts with usable closing totals | 6,031 |
| Printed final total in those parts | 4,626,568 |
| Closing tables with valid arithmetic | 6,031 |
| Timed-out sources | 2 |
| Completed sources without usable closing totals | 10 |

These are printed source controls, not recovered active electors or surname
coverage. The 4,317 zero-filled Urdu files remain unavailable. Two PDFs also
time out at 60 seconds; partial output is discarded. The exception review finds
five sources ending in PostScript error messages, two blank closing tables,
one closing table with its final total visibly blank, and two sources ending
before their closing tables. No missing total is inferred from other rows.
The final-page footer disagrees with the physical page count in 18 PDFs; the
closing-page pass does not check every intervening page.

Of 878 readable Hindi/Urdu filename-key overlaps, 875 have closing totals in
both languages and all 875 agree (`jk_2018_urdu_hindi_controls.json`). This is
control-total agreement, not proof that the elector records or editions are
identical. Do not add the two language totals or merge their rows by filename.
Five complete closing tables were visually checked: 120 numeric cells agree
with extraction. These source-format checks do not measure name accuracy.

The 56-part all-page pilot retains 43,260 source appearances, including
supplements and repeated entries, with 716 visible deletion stamps. Every
appearance has a serial; 90 lack an unambiguous Latin-font identifier. All
56 closing totals are available and sum to 41,487. An appearance count is not
an active-elector count: event classification, assembly/NPR eligibility and
names remain null. `jk_2018_urdu_pilot_audit.json` records this boundary.
One pilot PDF has 51 physical pages versus 52 printed, with the gap at physical
page 35. Three pilot parts have no checkable Latin page sequence.

Nafees reference-font versions 1.00–1.02 were obtained from the publisher and
hashed locally. The two sampled 982-glyph subsets each contain 252 nonempty
glyphs; only 200 match version 1.00 exactly. Reference shaping separates letter
bodies and dots, so individual outline matches do not establish Unicode text.
AC073 also uses TJ's Nastaleeq, a separate font family. Unmatched glyphs remain
unresolved. See `jk_2018_urdu_font_audit.json` for provenance and comparisons.
No LLM inference or Urdu name promotion has been performed.

```sh
python -m model_training.prep_er_data.jk_urdu_audit \
  --pdf-dir data/jk_recovery/raw/urdu_full \
  --split model_training/jk_2018_urdu_full_split.json \
  --out-dir data/jk_recovery/urdu_full_closing_verified \
  --workers 8 --timeout 30 --closing-only
```

Use a new output directory. The pilot uses `jk_2018_urdu_pilot_split.json`,
`--workers 4 --timeout 60`, and omits `--closing-only`. The original pilot's
source appearances were generated before the final closing-page overflow fix;
their controls agree with the final pass. Raw PDFs, font files, crops and
person-level appearances remain local. Public manifests contain aggregate
counts, source hashes and error dispositions.

The related upnaam improvement binds elector input/output paths as SQL
parameters, including its confidence-writing pass. Filenames containing
apostrophes now work through the complete artifact builder. This is covered by
a regression using both ordinary and quoted filenames. The English handoff below adds source validation and surname abstentions;
Urdu name/event recovery remains work; neither the published
lookup nor model has changed.

Local checks: 240 instate tests pass (one live test deselected; 97.76% coverage),
including 11 Urdu checks. Upnaam passes 223 tests with 92.50% coverage. Lint,
formatting, typing, docstrings and documentation builds pass in both packages;
upnaam's wheel and source distribution also build successfully.

### Hindi upnaam handoff, September 11

The reconciled Hindi inventory now has a native-script upnaam handoff for
historical AC057–AC080. It verifies inventory and PDF hashes, serial identity,
activity flags, event-key provenance and per-part counts before writing a complete
artifact. The aggregate audit is `jk_2018_hindi_upnaam_audit.json`.

| Handoff result | Rows |
| --- | ---: |
| Source inventory | 2,315,867 |
| Inactive rows excluded | 54,667 |
| Active NPR rows excluded | 30,229 |
| Active assembly rows retained | 2,230,971 |
| Corroborated native tokens selected | 1,161,254 |
| Abstentions | 1,069,717 |
| Relative-only candidates, kept separate | 24,524 |

All output keys match active assembly source records. Every selected raw token
occurs in the accepted own name, and independent membership checks confirm its
stated household or relationship evidence. The native tokenizer preserves vowel
signs and nukta, compares exact normalized spellings, and excludes initials,
titles and unsupported tokens. It has no uncorroborated position rule; 3,183
records abstain on conflicting evidence. Shared given names can still satisfy
corroboration, so the 52.05% selected share is not surname accuracy.

The handoff retains 55,143 rows whose source name is missing or has a parser issue.
It withholds a further 3,602 parser-accepted names under its input policy, mostly
because they contain colons or digits; these need source review. In total, 58,745
own-name fields, 51,471 relative-name fields and 11,012 house fields supply no
evidence. Only `पिता` and `पति` supply father and husband relationship evidence.
Raw candidates, labels, house values and event keys remain available locally.

Latin name fields, Latin surname fields, canonical surnames and confidence are
all null. Native selections use `surname_raw` and `surname_source_normalized`;
`latin_form_unavailable` records why they have no canonical Latin form. No
transliteration, Latin frequency-table update, model change or external LLM call
was made. Source count discrepancies and missing pages remain unresolved.

```sh
upnaam resolve-jk-hindi \
  data/jk_recovery/hindi_full_complete/inventory.parquet \
  data/jk_recovery/hindi_full_complete/audit.json \
  data/jk_recovery/hindi_upnaam_verified
```

The English adapter shares the inventory checks. A complete rerun produced zero
prepared-row or surname-row differences across all 169,191 English active
assembly records. Local tests passed: 242 upnaam tests and 240 instate tests
(one live test excluded), with formatting, lint, typing and docstring checks.

### English upnaam handoff, September 11

The audited English inventory now has an explicit upnaam handoff. The command
verifies its Parquet hash, per-PDF hashes and counts, activity flags, and unique
serials within assembly/NPR scope. It retains every active assembly row,
including missing-name rows, and publishes the output directory only after the
complete build succeeds. The aggregate evidence and code hashes are in
`jk_2018_english_upnaam_audit.json`.

| Handoff result | Rows |
| --- | ---: |
| Source inventory | 171,653 |
| Inactive rows excluded | 2,440 |
| Active NPR rows excluded | 22 |
| Active assembly rows retained | 169,191 |
| Corroborated written tokens selected | 66,636 |
| Abstentions | 102,555 |
| Relative-only candidates, kept separate | 2,338 |

The selected share is 39.39%, not an accuracy estimate. Selection requires
exact normalized spellings in household or explicit relative evidence; there
is no uncorroborated position fallback or spelling merge. Conflicting evidence
causes 812 abstentions. All confidence values remain null. The 39 missing-name
rows remain in the output; 379 relative-name fields provide no evidence.
`Self`, `Name`, `Son in Law`, `Adopted Son` and missing relationship labels do
not contribute relative-name evidence. Their raw labels remain available.

Every output key matches an active assembly source row. Every selected raw
token occurs in the accepted source name, and its normalized spelling remains
unchanged by household comparisons. Those checks establish source provenance,
not hereditary-surname accuracy: shared given names remain a concern. The
source's historical AC047–AC050 coverage, missing pages and count discrepancies
remain unchanged. This is a local research artifact, not a national lookup
update or a released replacement for the existing surname tables.

```sh
upnaam resolve-jk-english \
  data/jk_recovery/english_full_verified/inventory.parquet \
  data/jk_recovery/english_full_verified/audit.json \
  data/jk_recovery/english_upnaam_verified

python model_training/prep_er_data/name_tables.py lastnames-upnaam \
  --surnames data/jk_recovery/english_upnaam_verified/surnames.parquet \
  --lang jk_english_2018 \
  --out-dir data/jk_recovery/english_upnaam_verified/last_names
```

Use a new output directory. The intermediate count table contains 804 strings
with total weight 66,636; relative-only candidates contribute no weight. The
upnaam suite passes 232 tests with 92.59% coverage, including nine new adapter
checks. Lint, formatting, typing, docstrings, documentation and distribution
builds pass.

A local Tesseract Urdu header trial on eight development pages still misreads
component labels and numbers. No event classifications or names were accepted
from it. `history/jk_2018_urdu_header_ocr_trial.json` pins the model and output hashes.
Urdu name/event recovery and a validated Hindi surname handoff remain work.
No LLM calls have been made.

The Hindi archive contains 3,145 CSVs across AC057–AC080, including 12 empty
CSVs, plus one AppleDouble metadata file. It has 2,363,668 rows: 52,963 marked
deleted, 52,976 with blank names, 750,897 with U+FFFD in the name, and 256,422
with U+FFFD in the final token. These categories overlap. Summary extraction
also fails: 1,242 nonempty parts report zero electors, and one part reports
year `1`. Those summaries are not validated denominators.

The historical raw archive is at
`gs://in-electoral-rolls/jammu_kashmir_pdfs.tar.gz` (5,435,596,435 bytes;
generation 1521300993389689). The complete download matches its published MD5,
`c99b5c7ca2b4dc66486e0e8780f0ff4b`; its SHA-256 is recorded in the archive audit.
The initial bounded prefixes supplied the archived glyph and text pilots. Extraction
exposes legacy Mangal font substitutions as well as ordering problems; absence
of U+FFFD alone does not establish correctness. The 6,043 readable Urdu PDFs
contain 235,658 physical pages; their name extraction remains unfinished.

### Full Hindi rebuild results

The final artifacts are under `data/jk_recovery/hindi_full_complete/`; the compact
manifest is `jk_2018_hindi_full_audit.json`. All 3,146 input files
were attempted, with 3,145 parsed and the empty file explicitly failed. The
2,403,946 retained events yield 2,315,867 inventory identities: 2,261,200 active
and 54,667 inactive. Active inventory includes 2,230,971 assembly-eligible
records and 30,229 NPR records. Accepted names are available for 2,205,263 active
records; 55,937 remain unavailable, comprising 50,792 rendering failures,
4,666 source placeholders and 479 missing names.

Across all source appearances, wrapped-line handling changes 35,263 elector-name
candidates and 13,802 relative-name candidates. It withholds 738 previously
accepted elector appearances because the full continuation does not verify.
These counts include change events and inactive records, not just unique electors.

| Same-part comparison: 3,138 printed final totals | Original CSV, deletion flags removed | Recovered active inventory |
| --- | ---: | ---: |
| Records in comparable parts | 2,308,077 | 2,257,954 |
| Parts matching their printed final total | 295 | 3,076 |
| Net difference from 2,257,973 printed electors | +50,104 | −19 |
| Sum of absolute per-part differences | 69,320 | 145 |

The comparison excludes seven readable PDFs without a closing table. Their
3,246 active records remain in the inventory with no validated final denominator.
Sixty-two comparable parts remain discrepant. Across the corpus, 2,385 parts
match all component counts without recorded issues. There are 29 repeated
deletion events, 66 duplicate entry serials and nine page-sequence mismatches;
all 77 formerly unmatched changes now link after the NPR-heading repair. Two
readable PDFs have indeterminate page checks. Matching final counts does not
validate every correction or establish name accuracy.

The artifact checks passed for source and output hashes, schema agreement,
unique event and inventory keys, per-part counts, entry/change foreign keys,
identity consistency, explicit deletion status, and accepted-name damage guards.
No malformed source rows were silently skipped and no missing elector was
invented to match a total.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m model_training.prep_er_data.jk_hindi_rows \
  --pdf-dir data/jk_recovery/raw/hindi_full \
  --reference-font data/jk_recovery/reference_fonts/mangal11jun2009.ttf \
  --reference-font data/jk_recovery/reference_fonts/mangalb.ttf \
  --split model_training/jk_2018_hindi_full_split.json \
  --out-dir data/jk_recovery/hindi_full_complete \
  --workers 8
```

Use a new output directory when reproducing. The command recreates the event
ledger, inventory, schema and full audit; the compact manifest records the
additional old-CSV comparison and artifact validation. The repaired corpus is
not yet a released surname table. The native Hindi upnaam handoff is complete;
Latin transliteration and surname validation remain necessary for a lookup update.
Urdu text recovery, overlapping editions and missing sources remain open.

Downstream safeguards are repaired too. Instate rejects malformed CSV rows
in both single-name and two-name readers instead of allowing DuckDB to silently
skip them. Both readers accept Parquet and preserve paths containing apostrophes.
ASCII name aggregation also withholds replacement-character damage and includes
those records in its residual count. Upnaam's `normalization-v2`
keeps Latin normalization unavailable when a token contains U+FFFD, preserving
the damaged source. For example, `Sha�ma` no longer becomes `shama` through
ASCII conversion. This does not reconstruct letters or establish a surname boundary.

The English rebuild and PDF-to-CSV regression pass. Historical test counts
remain in the archived audit records. No J&K lookup or model artifact has
been promoted from this audit.
