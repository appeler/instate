# Historical recovery experiments and release preparation

These are superseded development notes. Current source counts, handoffs and
remaining work are in [the recovery checkpoint](../coverage_recovery_plan.md).
Audit files retain the paths and hashes recorded when the experiments ran.
Commands below run from the repository root.

### Hindi glyph pilot, first pass

The embedded Mangal fonts lack the full character and shaping tables needed
to decode missing mappings. The pilot hashes exact decomposed glyph outlines
and reuses a Unicode mapping only when that outline has a unique usable source
mapping in the six development PDFs. It preserves existing usable mappings,
rejects conflicting evidence and leaves unsupported glyphs unresolved. Font
names or glyph numbers alone never establish equivalence across documents.

The frozen development/holdout split is in `jk_2018_hindi_pilot_split.json`;
source hashes, per-part results, software versions and output hashes are in
`jk_2018_hindi_glyph_pilot.json`. The development catalogue has 180 mappings
and no conflicting assignments.

| Physical Mangal glyph groups | Development: 6 PDFs | Holdout: 20 PDFs |
| --- | ---: | ---: |
| Usable source mappings preserved | 349,145 | 1,219,469 |
| Missing/invalid mappings recovered | 1,866 | 11,169 |
| Unresolved after repair | 7,071 | 23,265 |

This recovers 32.4% of the 34,434 damaged holdout glyph groups. These counts
include headers, labels and other text, not just names; they are not recovered
elector counts or accuracy estimates. Inspection of 20 distinct, fully recovered
holdout spans against source crops found no glyph disagreement. That conditional,
single-reviewer inspection does not estimate general accuracy.

The output preserves source text, physical glyphs, positions and decoded visual
order. It still needs logical Unicode ordering: a visually printed `शर्मा` can
produce `शमार्` when decoded in drawing order. That artifact cannot supply surname
training inputs. Government-hosted Mangal versions 0.99 and 1.20 did not match
the embedded outlines. The matching-font requirement was resolved in the second
pilot below; no font path is still required from the user.

To reproduce with the complete local pilot PDFs and a new output directory:

```sh
uv pip install --python .venv/bin/python pymupdf==1.28.2 fonttools==4.65.0
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python model_training/prep_er_data/hindi_glyphs.py \
  --pdf-dir data/jk_recovery/raw/gcs_pilot \
  --split model_training/history/jk_2018_hindi_pilot_split.json \
  --out-dir data/jk_recovery/hindi_pilot/outline_v1
```

The script writes `catalog.json`, `spans.jsonl.gz` and `audit.json`. Raw text and
source crops remain local under the git-ignored `data/` directory. Any decoder
changes informed by the current holdout require new held-out parts for evaluation.

### Hindi text recovery, second pilot

Mangal 5.90 regular and bold from the
[Rajbhasha download archive](https://rajbhasha.net/mangal-fonts-download/)
match the source outlines. Their full character and GSUB tables resolve the
missing mappings without OCR or an LLM. Font filenames, versions, archive and
font hashes are recorded in `jk_2018_hindi_text_audit.json`. The font binaries
remain local and are not distributed with this repository.

`hindi_text.py` derives candidates from the modern Devanagari default and
language-specific font substitutions. It identifies reph through the `rphf`
feature rather than treating every `र्` as a movable mark. Conflicting meanings
propagate through substitutions and remain withheld: 1,265 exact-outline mappings
are usable, and two ambiguous outlines are excluded. It reconstructs words from
individual glyph positions because a PDF span can contain characters from
different words or lines. Pre-base `ि` and tagged reph are reordered within
consonant clusters. Each candidate is then reshaped with HarfBuzz; `logical_text`
is populated only when the resulting outlines match the source sequence exactly.
The font-defined candidate remains separately available when that check fails.

The original six development PDFs remain development data; the original twenty
holdout PDFs are now regression data. Another bounded 32 MiB of the same archive
generation supplied 21 complete, previously untouched Hindi PDFs for validation.
The full split is frozen in `jk_2018_hindi_text_split.json`; PDF content hashes
are also checked to reject duplicate sources across groups.

| Text recovery measure | Development: 6 PDFs | Regression: 20 PDFs | Validation: 21 PDFs |
| --- | ---: | ---: | ---: |
| Unsupported source glyph groups | 8,937 | 34,434 | 36,477 |
| Recovered from reference outlines | 8,892 | 34,373 | 36,389 |
| Unresolved glyph groups | 45 | 61 | 88 |
| Reconstructed words | 75,817 | 268,043 | 271,309 |
| Words passing exact rendering check | 75,543 | 267,153 | 270,406 |

The new validation recovery rate is 99.76% of unsupported glyph groups. All 194
unresolved glyph groups across the 47 PDFs are source glyph zero (`.notdef`).
The same twenty regression PDFs improved from 11,169 to 34,373 recovered glyph
groups. These are all-Mangal-text counts, including headings and labels. They are
neither elector counts nor estimates of name accuracy or intended spelling.
Whitespace contributes to glyph totals but separates output words; non-Mangal
text is outside this decoder. The 903 validation words that fail rendering
verification remain unavailable as `logical_text`.

A synthetic ordering check covers 9,916 regular/bold consonant, vowel, reph and
conjunct cases: 9,700 return the original logical text, 216 abstain, and none of
the accepted cases changes that text. The 47-PDF extraction took 36.6 seconds
locally, excluding download time. This supports software correctness, not a
population accuracy estimate. Reference semantics follow Microsoft's
[Devanagari shaping specification](https://learn.microsoft.com/en-us/typography/script-development/devanagari).

```sh
uv sync --group roll-recovery
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m model_training.prep_er_data.hindi_text \
  --pdf-dir data/jk_recovery/raw/gcs_pilot \
  --pdf-dir data/jk_recovery/raw/gcs_validation \
  --reference-font data/jk_recovery/reference_fonts/mangal11jun2009.ttf \
  --reference-font data/jk_recovery/reference_fonts/mangalb.ttf \
  --split model_training/history/jk_2018_hindi_text_split.json \
  --out-dir data/jk_recovery/hindi_pilot/text_v2
```

Use a new output directory. `words.jsonl.gz` retains raw text, decoded candidates,
accepted logical text, bounding boxes, physical glyphs, outline fingerprints and
per-glyph status. `audit.json` records source/software/output hashes and per-part
counts. Current artifacts remain under `data/jk_recovery/hindi_pilot/text_v2/`.

The subsequent row pilot below reconstructs elector events using card geometry
and all text fonts. The development probe found that rectangular frames also occur in
summary tables, so rectangle counts alone are not elector counts. The source
also draws `DELETED` letters in a different content-stream order; that literal
stamp reads correctly when its glyphs are ordered by horizontal position.
Do not force row counts to match a summary or infer missing `.notdef` characters.
No LLM calls were needed; if a subsequent pass needs an LLM, the user's selected
provider is Muse Spark Contributor.

### Hindi elector ledger and total reconciliation

`jk_hindi_rows.py` now rebuilds the 47 local pilot PDFs into two typed Parquet
artifacts: every visible base/addition/deletion/correction appearance in
`events.parquet`, and the resulting identity inventory in `inventory.parquet`.
`SCHEMA.json` documents types, nulls and field meanings directly from the parser;
`audit.json` records source hashes, software versions, output hashes, per-page
counts and printed summary arithmetic. The reproducible checkpoint is
`jk_2018_hindi_rows_audit.json`.

The old parser combined additions, deletions and corrections as supplementary
rows. The recovered ledger preserves their event types. Corrections update an
existing record without increasing the count; a deletion list and a `DELETED`
stamp on its base card are two pieces of evidence for one deletion. The stamp's
glyphs must be read in spatial order. Valid comma-grouped serials such as `1,257`
are normalized for matching while `number_raw` preserves the printed form.

Inventory identity combines serial and source assembly eligibility within a PDF.
NPR headings carry forward across continuation pages. Missing serials, duplicate
entry serials, unmatched changes and conflicting deletion IDs are explicit
issues; the parser does not fabricate a link. Card sides can survive when their
horizontal borders are missing, and parallel border strokes must not duplicate
cards. Empty template cards are excluded; numbered blank-name cards are retained.
Wrapped elector and relative names retain subsequent lines until the next labeled
field; every included line must pass rendering verification. On the pilot, this
changes 523 elector-name candidates and 179 relative-name candidates. Nine active
names previously accepted from their first line are now withheld because their
continuation does not verify.

The table retains the original text-pilot group names for traceability. All 47
parts now serve as regression cases for the row parser; these row results are
not a fresh held-out accuracy evaluation. The earlier text-decoder measurements
refer to the frozen decoder and its original evaluation.

| Measure | Development: 6 parts | Regression: 20 parts | Validation: 21 parts |
| --- | ---: | ---: | ---: |
| Source events retained | 4,453 | 15,756 | 16,000 |
| Active inventory, including NPR | 4,059 | 14,797 | 15,192 |
| Active assembly-eligible inventory | 3,863 | 14,437 | 15,072 |
| Active NPR inventory | 196 | 360 | 120 |
| Active records with accepted names | 3,955 | 14,438 | 14,846 |
| Parts matching printed final total | 5 | 20 | 21 |
| Parts matching every component count | 3 | 14 | 17 |
| Parts with identity-linking issues | 0 | 0 | 0 |

Across all 47 parts, 36,209 events yield 34,833 inventory identities, including
785 inactive identities. Active inventory is 34,048 against 34,047 printed:
46 parts match exactly. Of the active records, 33,239 have names passing the
rendering and placeholder checks; 809 retain unavailable names with source/candidate
evidence. This includes 68 names with a printed dotted-circle placeholder.
Those are extraction decisions, not verified surname boundaries or accuracy rates.

The original CSVs have 35,687 rows, including 774 marked deleted. Simply removing
those flags leaves 34,913 rows, 866 above the newly recovered printed totals;
only four parts match. They also have 982 repeated serial-and-ID appearances and
16 zero-valued reported totals. Repeated appearances are a diagnostic, not a
license to deduplicate without applying the source events. The new final-count
net and absolute discrepancies are both one.

For HACA059PS0025, the source ledger is 501 base records, 35 additions, 26 deletions
and 53 corrections: 510 active electors. The old CSV has 539 rows after removing
deletion flags. HACA075PS0060 retains 1,118 active records against 1,117 printed;
it visibly lists 13 additions but reports 12. This difference remains intact.
Twelve other parts have fewer recovered correction events than their closing
tables report, despite matching final elector totals. These component differences
remain flagged; matching the final count does not validate every name update.

```sh
uv sync --group roll-recovery
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m model_training.prep_er_data.jk_hindi_rows \
  --pdf-dir data/jk_recovery/raw/gcs_pilot \
  --pdf-dir data/jk_recovery/raw/gcs_validation \
  --reference-font data/jk_recovery/reference_fonts/mangal11jun2009.ttf \
  --reference-font data/jk_recovery/reference_fonts/mangalb.ttf \
  --split model_training/history/jk_2018_hindi_text_split.json \
  --out-dir data/jk_recovery/hindi_pilot/rows_current
```

The output directory must be new. Retain `active` and `assembly_eligible` as
distinct fields when selecting a population. Raw age, house and sex candidates
are not recoded from inferred values. Apply no surname inference to unavailable
names. Residual component differences need attribution and Urdu name extraction
remains unfinished. Full-state surname aggregates remain unchanged.

### Repairs identified in the full Hindi corpus

The frozen full split is `jk_2018_hindi_full_split.json`. The reader uses separate
worker processes and writes each part to a Parquet row group, with explicit
per-file errors. The empty `HACA069PS0044.pdf` is a source failure, not a
zero-elector part. Page-number checks compare printed footers with physical
pages; missing footers leave the check unknown instead of establishing completeness.

Some component headings contain a missing glyph inside `घटक`. The fallback
requires an explicit component number and its corresponding title—addition,
deletion or correction—with rendering verification. It does not infer the
missing character. NPR detection requires the parenthesized assembly-voting
restriction; `एन पी आर` in a locality name cannot change eligibility. Seven
parts had 77 changes that failed to link under the broader locality match.

Repeated deletion-list entries remain separate events and are flagged; they
deactivate an identity once. Source defects remain visible: HACA071PS0063 repeats
30 base records from physical page 12 on page 14, with matching IDs, names,
relative names, houses, ages and deletion stamps. HACA071PS0131 skips printed
page 49; its next physical page prints page 50 and a subtotal of 14 additions.
No addition cards survive in that PDF, consistent with an omitted addition page.
The missing records cannot be reconstructed from the subtotal.

## 1. Relative-name evidence in upnaam

Keep selection from the elector's written name separate from a candidate borrowed
from a relative. Remove unconditional inheritance from the recorded-surname ladder.
Existing corroboration, conflict abstention and the Karnataka initials exception
take precedence. A candidate needs a known relationship, a usable elector name,
and either an exactly matched co-resident with a corroborated recorded surname or
an explicitly supplied relative-name position policy. Do not infer a surname from
a one-word relative name, resolve a conflict by inheritance, or propagate inferred
values to another person. Preserve source spelling, relationship and evidence;
leave accuracy unmeasured. Test the rules and the elector artifact together.

## 2. Coverage adjustment in instate

Use the separate coverage module now under an explicit MCAR assumption within
each state and roll edition. Divide the chosen target count by the observed
surname count and multiply every surname count in that population by the result.
Return observed counts, weights and adjusted estimates separately. Do not alter
source counts or silently replace the model's training or lookup artifacts.

Use parsed elector totals to adjust surname non-resolution; use comparable roll
totals when also assuming missing source records are random within the state.
Record that denominator choice and the assumption in each run. Exclude people
known not to use a family surname; treating other unresolved names as missing
surname values is an explicit modeling assumption, not an audit prerequisite.
Do not mix editions or historical geographies. A positive-target state with no
observed surnames still cannot be weighted. Missing constituencies inside an
otherwise observed state can be covered by the state-level MCAR assumption;
this is extrapolation, not recovered source evidence.

Follow the concise [weighting plan](../docs/coverage.md#weighting-plan): produce
the MCAR baseline, retain its diagnostics, then evaluate finer strata or reference
recovery as improvements. Ration/land linkage, a reference audit and held-out model
evaluation are follow-up work, not prerequisites for the first weighted estimates.
Test mass conservation, preservation of observations and invalid-input rejection.

## 3. Jammu and Kashmir and Ladakh

Inventory local parsed and raw sources, establish edition and language coverage,
and compare extraction with printed part totals. Separate missing Urdu source
extraction from downstream surname failures. Repair supported local extraction
paths, evaluate on untouched parts, and retain the combined historical geography.
Do not treat a successful English/Hindi parse as evidence of Urdu coverage.

## 4. Dadra and Nagar Haveli

Reconcile input records, relative-name columns, selected surnames and retained
counts. Check whether separate father/husband/mother columns are being combined
without keeping relationship type, and whether available source parts are lost.
Keep Dadra distinct from Daman and Diu for the historical model vocabulary.

## 5. Andaman and Nicobar Islands

Reconcile source parts and records through parsing, normalization, surname
selection and minimum-cell filtering. Inspect extraction failures against source
records and validate fixes on separate parts before rebuilding aggregates.

For every state, record measured counts and remaining external requirements.
Source recovery, surname candidates and coverage weighting have separate totals.
Run local tests and project lint/type/docstring checks after code changes. A
rebuild or release is not justified by an increased surname fill rate alone.

## Implementation status

Surname resolution now lives in `../upnaam/src/upnaam/ladder.py`; its contract and
validation design live in `../upnaam/docs/surname-ladder.md`. The first implementation
passes 220 local tests (92.45% coverage), Ruff, Pyright, pydoclint, documentation,
and wheel/source builds. These are software checks, not empirical accuracy estimates.
Reference-data locations were requested for independent surname validation.

Coverage adjustment is implemented in `src/instate/coverage.py`, documented in
`docs/coverage.md`, and is not enabled in runtime inference or training. The full
local suite passes with the matching pinned model artifacts supplied through
`INSTATE_MODEL_DIR`: 127 tests, 97.76% total coverage, 100% for the new module.
The artifact hashes were checked against `_resources.ARTIFACT_SHA256` first.

## Small-state parser work

The first repairs target `../parse_searchable_rolls`, not surname inference:

- Shared XML text extraction: replace the removed `getchildren()` API with
  `itertext()` and preserve text after inline tags, including surname suffixes.
- Andaman: recognize mother-name labels when extracting the elector's name;
  restrict district extraction to district names rather than numbered sections.
- Dadra: retain the house number when the adjacent ID is the documented `NILL`
  marker or `NIL`, without treating those strings as valid elector identifiers.
  The raw ID field is preserved rather than invented.

Regression fixtures cover relationship boundaries, inline XML text, district
misidentification and missing-ID house extraction. Generated PDFs exercise the
installed Poppler converter and both complete readers; these are synthetic tests,
not evidence of a measured increase in corpus coverage. Run them from the parser
repository with `python -m pytest` and Poppler `pdftohtml` installed.

The existing Dadra English report has 217,840 parsed rows, versus 216,272 records
of mass in the current instate name aggregate. The 1,568 difference is not yet an
attributed parsing loss: the transliteration configuration also identifies a
Gujarati source, so source edition/language, active-record filtering and
romanization must be reconciled before comparing those totals as pipeline stages.
Andaman's English report and current name aggregate both have mass 215,815.

An authenticated top-level listing of `gs://in-electoral-rolls/` found neither
`andaman.tar.gz` nor `dadra_pdfs.tar.gz`; the documented Andaman object returns
404. The bucket does contain `jammu_kashmir_pdfs.tar.gz`. Located local repositories
contain the parser code, reports and a Dadra token archive, but no matching source
rolls were found in the searched data directories. Dataverse sources were subsequently located.
The user subsequently approved Dataverse access. The Andaman source archive
(file 6893972, 94,533,551 bytes) is now downloaded under the git-ignored
`data/small_state_recovery/raw/` directory. It contains 401 PDFs. A bounded 32 MiB
prefix of Dadra archive file 6898917 supplied three English PDFs without requiring
the complete 3.5 GB split archive. The later Andaman rebuild covers its complete
archive; Dadra remains a bounded source sample, expanded to 128 MiB for validation.

### Initial source pilot (superseded by the rebuild below)

`data/small_state_recovery/andaman_pilot.json` records archive/member hashes and
original-versus-fixed state-pattern comparisons on parts 1, 2, 100, 200, 300 and
354. Both comparisons use the XML compatibility repair so they can run on current
Python. The first state-pattern fixes recover 20 rows, increasing the six-part
total from 3,312 to 3,332. Part 354's district changes from the section label
`2-GANDHI STATUE` to `NICOBAR`; part 100's previously empty district is recovered.
This is a diagnostic sample, not a statewide coverage estimate.

The initial remaining Andaman gaps had concrete causes. In part 100,
146 of 884 main-roll boxes are rejected; in part 200, 189 of 685 are rejected.
Missing names account for 145 and 189 respectively. Dotted initials occur in
144 and 181 rejected name-containing fragments. The name regex excluded
periods. Part 354 also had 12 rejected boxes with missing elector numbers.
Subsequent repairs retain written names, recognize `Other` relationship labels
and handle the printed `#` marker without losing elector numbers.

`data/small_state_recovery/dadra_pilot.json` records the bounded-download provenance
and comparisons. Main part 116 produces 915 rows against a parsed printed total of
919 under both versions; neither version reads the general information in standalone
supplements 074 and 128. Supporting and reconciling those supplements is a higher
priority than simply expanding the existing counts. No observed row gain is claimed
for the Dadra missing-ID house-number fix in this sample.

### Installed repairs and full Andaman rebuild

The fixes are installed in `../parse_searchable_rolls`; all 60 local parser tests
pass, including PDF conversion and CSV export tests. Lint passes for the changed
parser and test code. The temporary XML-helper installation error is corrected.

The repairs also retain additions with blank house numbers and distinguish main
rolls from supplementary layouts. Andaman part 143 restarts serial numbering:
289 colliding records are now retained rather than overwritten. Corrections target
a unique EPIC first, then a unique serial; ambiguous matches are counted, not used
to overwrite an arbitrary record. No unmatched corrections remained in this rebuild.
CSV output preserves the source `number` and adds `relative_type`. A number alone
is not a unique record key within a PDF. Existing CSVs with the old header are
rejected for appending; write a new output file rather than mixing schemas.

`data/small_state_recovery/andaman_rebuild_v2.json` records the final per-part
comparison and parser hashes. The earlier rebuild manifest supplies the full
source-member inventory and hashes. Both sides of the comparison use the same PDF
text extraction, with the original state patterns and original record handling
used for the baseline.

| Measure | Original parser baseline | Repaired parser |
| --- | ---: | ---: |
| PDFs parsed | 401 | 401 |
| All parsed rows | 215,815 | 280,101 |
| Active rows, excluding marked deletions | 213,999 | 277,978 |

The repaired parser recovers 64,286 rows, including 63,979 additional active rows.
Printed active totals sum to 277,983. Of 401 parts, 393 match exactly; five are
one above their printed totals and three are below by 1, 1 and 8. Thus the net
shortfall of five masks 15 absolute count discrepancies across eight parts.
Those residuals remain explicit; no weighting or invented records erased them.

The new raw-row artifact is
`data/small_state_recovery/parsed/andaman_2017_repaired.csv`. Source and person-level
artifacts are git-ignored. Existing surname aggregates, lookup and model artifacts
are unchanged. Surname resolution on these rebuilt records remains in `upnaam`.

### Dadra independent source validation

`data/small_state_recovery/dadra_validation.json` reports 20 source PDFs, including
17 not used to develop the fixes. All 14 main-roll PDFs match their printed totals.
All six supplements match their printed addition/deletion event counts: 204
additions and three deletion entries. One deletion entry repeats in the source;
the parser retains it and reports the duplicate rather than claiming three unique
people were deleted. The full 3.5 GB Dadra archive has not been rebuilt.

Diagnostic and independent-sample CSVs are under
`data/small_state_recovery/parsed/`. Supplement rows are change events, not a
complete electorate. Reconcile them with matching main rolls before updating
state surname counts or using their totals for MCAR adjustment.

## Small-state baseline and source requirements

The aggregate baseline and input hashes are recorded in
`small_state_recovery_baseline.json`. These are existing local artifacts; no
raw elector source or surname accuracy is inferred from their fill rates.

| Source label | Name-table record mass | Surname-table record mass | Interpretation |
| --- | ---: | ---: | --- |
| J&K | 164,713 | 2,185,245 | Not comparable stages: the name table matches the historical English-only report, while the surname table contains additional mass. Reconcile the combined source manifests before calculating retention. |
| Dadra | 216,272 | 196,317 | Apparent 90.8% retention from local names to surname counts; verify common source edition and scope before treating this as a pipeline rate. |
| Andaman | 215,815 | 205,466 | Apparent 95.2% retention on the same limited basis. This does not measure coverage against the full electoral roll. |

Proceed in order: Andaman part/record completeness and remaining parsing losses;
Dadra source reconciliation and relationship-column handling; J&K source-manifest
reconciliation and Urdu extraction. The located parser repositories contain
code and descriptive reports, but raw PDFs/parsed elector source locations still
need to be supplied. Those paths were requested from the user. Do not invent
source-based validation, infer statewide coverage from these aggregate tables,
or apply weights merely to make totals match.

## September 10 implementation and release preparation

This section supersedes earlier small-state pilot counts.

- Andaman: all 401 PDFs rebuilt; 280,110 retained rows, including 277,987 active
  records against 277,983 printed. Exactly 395 parts reconcile; the six residual
  differences have source-side explanations. The latest parser repair recovered
  nine supplement rows and one source-backed correction. The parser suite passes
  68 tests.
- Andaman deposit: 182,787 recorded surname selections and 27,171 relative-only
  candidates, kept separate. MCAR factor 1.5208029017380886 uses printed electorate
  totals and does not enter training. The 30-record crop sample and archive hashes
  were checked. Deposit: `data/andaman_2017`.
- Dadra: complete raw archive downloaded and checksummed; 266 English main PDFs
  and 255 English supplement PDFs. All main-roll counts match, totaling 217,934.
  English supplements are missing for parts 24, 30, 38, 40, 93, 94, 132, 159, 222,
  224 and 251. Do not silently mix available January-final events into the complete
  draft frame.
- Dadra draft deposit: 155,108 recorded surname selections; surname-first fallback
  after explicit relation evidence. MCAR factor 1.4050468060963974. All 30 crop
  samples checked, including a duplicated EPIC disambiguated by printed serial.
  Deposit: `data/dadra_2017`.
- Dadra supplements: 9,110 source events, 9,089 applied and 21 retained duplicates.
  Three initially ambiguous EPIC matches are resolved using the printed serial.
  The 255-part final subset has 217,357 active records versus 217,348 printed;
  residuals reflect repeated deletion entries. Separate event, control and
  reconciled-record Parquets are under `data/small_state_recovery/parsed`.
- Karnataka cleanup: removed 12 bytecode caches and three abandoned pending
  shards after validating their completed replacements; 28,568,388 bytes removed.
  Canonical deposits and unique historical evidence retained. Ledger:
  `data/small_state_recovery/karnataka_cleanup.json`.
- Candidate 3.3.0 national tables: only Andaman and Dadra source inputs change;
  the other 33 hashes are unchanged. There are 1,855,202 surname strings,
  2,793,924 retained cells and 737,984,272 retained observations. Lookup and
  training cells match exactly, with a minimum of three observed records.
- Eight-epoch developmental retraining is running under `data/release_3_3_0`.
  The historical test is not treated as untouched. No model or package published.
- Local instate suite: 141 passed, one live test deselected, 97.76% coverage.
  Lint, format, typing and docstrings pass. Publication still requires completed
  training/calibration, independent review and the release workflow.

### Release checkpoint decision, 2026-09-10

Training and calibration have finished. This status supersedes the earlier
training-in-progress notes. The native SafeTensors checkpoint was staged without
conversion: its bytes, all 19 tensors, and comparison logits matched exactly.

Do not promote the retrained weights in the proposed 3.3.0 data/weighting release.
On the same 20,000 hash-ranked validation names with the updated observed-count
targets, the existing 3.2 checkpoint has lower record-weighted log loss and Brier
score, and higher top-three record mass:

| Checkpoint | Log loss | Brier score | Top-three record mass |
| --- | ---: | ---: | ---: |
| Retained 3.2 | 1.373 | 0.235 | 79.2% |
| Held retrained candidate | 1.434 | 0.263 | 78.1% |

The paired surname-bootstrap interval for the log-loss difference is -0.016 to
0.131. It includes zero; retaining the existing model is a conservative release
decision, not proof of a population-level performance difference. Karnataka's
regional result improves, but that does not override the overall selection
criterion. These are developmental comparisons, not untouched-test evidence.

The candidate's completed calibration uses 165,701 other validation names and a
temperature of 1.208. Its checkpoint and calibration remain together as a held
experiment. The proposed release instead retains both the 3.2 checkpoint and its
matching calibration. Its updated lookup has a different source revision from
the retained model's training data; the release manifest records that distinction.
Neither the lookup nor the model receives MCAR weights.

The local combination of the repaired lookup and retained model passes the
runtime examples, the documented MCAR example, and all 141 default tests (one
live-service test deselected), with 97.76% coverage. Formatting, lint, type checks,
and docstring checks pass. The 28 coverage/export/reconciliation tests also pass
in an isolated code-only snapshot.

Remaining release gates are independent review and finding dispositions, final
metadata and immutable artifact pins, distribution checks, CI, and publication.
The proposed 3.3.0 release has not been published. Local diagnostics, artifact
hashes, and the retention decision are under `data/release_3_3_0/`.
