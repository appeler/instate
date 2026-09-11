# Surname evidence and state coverage recovery

Relative-name evidence and isolated coverage adjustment are implemented. Current
parser execution order: Andaman and Nicobar Islands, Dadra and Nagar Haveli, then
Jammu and Kashmir and Ladakh.

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
