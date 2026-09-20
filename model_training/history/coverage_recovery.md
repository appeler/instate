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
instate release. Muse Spark 1.3 Contributor completed an independent code review of
commit `3a56730`, finding no concrete correctness defect. Its fresh source tests,
lint, formatting, typing, docstrings, syntax, documentation and build checks pass.
The review covers preparation code, not unpublished data or a final release.
GitHub CI passes at the same commit. No release tag, runtime-artifact promotion
or publication has occurred.

The Urdu decoder now reconstructs font-defined letter bodies and dots, retaining
all candidates that reproduce the exact source outlines. On physical page 3 of
the frozen pilot, it finds unique text for 106 of 1,723 development span segments
and 1,478 of 34,255 validation segments. These are PDF span fragments, not complete
names. Unmatched outlines remain the main obstacle; approximate shape matches are
diagnostic only. No Urdu name has been promoted into an elector artifact.
The decoder also passes all 1,806 synthetic single-letter and two-letter
roundtrips: 1,668 are unique and 138 retain multiple indistinguishable spellings.

Hindi transliteration review inputs are staged locally: 5,857 distinct native
selections and a frozen 152-token diagnostic pilot. The historical eroll spelling occurs among indicate candidates for
2,838 strings representing 1,134,578 selected occurrences; 1,307 strings occur in
only one corpus, 1,679 are missing, 31 have invalid Latin candidates, and two have
conflicting candidates. These are candidate coverage measures, not validation.
Of those matches, 1,321 strings (118,067 occurrences) have a single matching
candidate; 1,517 strings (1,016,511 occurrences) have alternatives. The earlier
`corpus_agreement` audit label means candidate-set overlap, not unique agreement.

The September 12 Muse Spark 1.3 Contributor diagnostic returned all 152 frozen
Hindi tokens without seeing historical Latin candidates. Seventy-five primary
answers match eroll, 18 list its answer as an alternative, 33 differ, and 26 have
no eroll candidate. Fourteen responses flag unusual spelling and one flags
uncertainty. Three rows contain single-letter alternatives: valid under the
prompt's ASCII rule, but quarantined under the stricter local two-letter rule.
These comparisons cover 1,071,393 selected occurrences but are not an accuracy
estimate. The high-frequency case `सिहं` remains unresolved: exact font evidence
on six development pages reproduces that literal spelling in all 67 inspected
instances, rather than the standard `सिंह`. Source spelling and a conventional
Latin name are separate questions. No mappings are promoted.

The Hindi calls cost an estimated $0.0034201 from reported usage. Full responses,
row comparisons and the cost ledger remain under
`data/jk_recovery/muse_review/`; the aggregate is in `state_handoff_audit.json`.
A separate, blinded 16-glyph Urdu diagnostic recognized one of eight exact-reference
controls and abstained on seven. It returned all labels and valid Unicode, but
failed the recognition gate. Its estimated cost was $0.0004943. No glyph mapping
is accepted, and isolated-glyph inference will not be scaled. Word grouping and
detached marks need further font-based diagnosis. Results are recorded in the
existing `jk_2018_urdu_font_audit.json`.

The Hindi handoff now accepts a local `--romanization-map` after native selection.
It rejects duplicate normalized keys and invalid Latin tokens. Transliteration
cannot change the native evidence, select a new token, or resolve an abstention.
The full development run applies 4,145 unambiguous, eligible historical-corpus
mappings: 1,149,498 selections receive Latin forms and 11,756 remain unmapped.
All 2,230,971 rows and every native evidence field are preserved; the prepared
electors are byte-identical. These spellings still require linguistic review.
Upnaam passes 261 source and installed-wheel tests at 92.80% coverage, plus lint,
typing, docstrings, documentation and distribution checks. The installed-wheel
English full rerun and Hindi six-part pilot reproduce their comparison artifacts
byte for byte. The prior source snapshot and tested wheel are retained locally.

The Hindi Latin count table has 3,636 strings and 1,149,498 occurrences. Combined
with English's 804 strings and 66,636 occurrences, the development J&K input has
4,267 strings and 1,216,134 occurrences. Their constituencies are disjoint:
English AC047-AC050 and Hindi AC057-AC080. Urdu is still excluded.

A national development build with these J&K counts and the corrected Telangana
input contains 1,823,593 surname strings, 2,720,702 retained cells and 729,145,998
occurrences. J&K retains 1,213,390 occurrences; Telangana retains 16,091,518.
Other states together lose 711 occurrences relative to 3.3.0 through national
spelling reconciliation. Counts agree with the training table, shares sum to one,
and retained cells meet the support floor. This is an integration check, not a
release candidate: Hindi spellings are unreviewed and Urdu remains incomplete.

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
At this September 11 checkpoint, no LLM inference had been performed. The later
diagnostic is recorded above; no Urdu name has been promoted.

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
This September 11 checkpoint predates the model diagnostics recorded above.

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
Sixty-two comparable parts remain discrepant. The residual triage groups 21 parts
with repeated deletion events (29 absolute difference), eight with duplicate
entry serials (68), one with a page gap (14), and 32 with component-control
differences but no ledger issue (34). These groups describe observed patterns,
not verified causes for every part. In development part HACA075PS0060, visual
inspection of physical pages 42–43 confirms 13 distinct addition cards, while
the component and closing summaries each report 12. All 13 source-supported
rows remain; the +1 discrepancy is documented rather than forced away.
Among 66 duplicate-serial groups, 62 share a nonempty ID and four have different
IDs. Eight header crops confirm the latter four collisions in the source. The
current ledger keeps the first entry and flags the duplicate serial; those
conflicting identities need preservation or explicit quarantine before release.
They must not be treated as confirmed duplicate electors.
Across the corpus, 2,385 parts
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
