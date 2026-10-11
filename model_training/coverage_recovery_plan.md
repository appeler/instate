# Surname evidence and state coverage recovery

The J&K recovery entered the 3.4 release and remains in the 3.6 lookup. Earlier experiments and
superseded checkpoints are in [the historical record](history/coverage_recovery.md).
All J&K work uses 2018 electoral rolls. Candidate affidavits are excluded.

## Final J&K source handoffs

The English inventory contains 169,191 active assembly records from historical
AC047–AC050. Upnaam selects 66,636 Latin surname occurrences and abstains on
102,555 records. Its source and output hashes are recorded in
`jk_2018_english_upnaam_audit.json`.

The Hindi inventory contains 2,230,975 active assembly records from historical
AC057–AC080. Upnaam selects 1,161,257 corroborated native occurrences and
abstains on 1,069,718 records. The reviewed map retains 913,449 Latin selections
from 3,503 native forms. Source warnings, insufficient corroboration and the
unresolved spelling `सिहं` remain withheld. The source/parser audit and handoff
audit are `jk_2018_hindi_full_audit.json` and
`jk_2018_hindi_upnaam_audit.json`.

The calibrated Urdu inventory contains 4,690,023 records, including 4,617,281
active records and 4,608,102 active assembly records. It covers 6,014 row-bearing
PDFs after 24 redundant source copies are excluded. The recovery accepts
2,315,587 own-name fields and 3,401,536 relative-name fields. Upnaam selects
970,947 corroborated native surname occurrences from 4,399 distinct selected
tokens and abstains on 3,637,155 records. Every selected token has a retained
Latin mapping. Unsupported fields remain missing and unsupported records remain
abstentions.

The calibrated recovery transfers a missing Urdu field only when at least six
independent drawings support one reading, the winning reading has at least 75%
support among both distinct records and calls, and it leads the runner-up by at
least one observation. On a hidden control set, 1,259 of 1,322 words matched
exactly (95.23%). This is a source-reading diagnostic, not a surname-accuracy or
population-coverage estimate. The promoted inventory changes no immutable
source field and introduces no duplicate key.

The final Urdu map contains 27,221 native/Latin pairs. Its SHA-256 is
`8f1c8f8211f001675678354cc1736c14b4a05ff9a971ba390c039e43fbc892e7`.
The shared compressed corpus has SHA-256
`35be72079f8a226db1e9794ac91fa3c9f921fc673287c93450f23514502cd373`.
Indicate rebuilds a 27,221-key local lookup from this corpus; the installed table
hash is `570206fa75d4c12169bd7d09aa081ce0b268f8208d50fa7c5aaa622c5f862950`.

## Edition reconciliation

Hindi and Urdu are overlapping editions, not disjoint populations. Exact source
identity, constituency, part, serial and record-ID matching finds 693,201
one-to-one card links. The final table counts each linked card once, preferring
the mapped Hindi selection when both editions select a surname and otherwise
using the mapped Urdu selection. This yields 913,449 Hindi selections, 967,686
Urdu selections after overlap reconciliation, and 66,636 disjoint English
selections: 1,947,771 J&K source occurrences across 6,538 Latin strings.

The link audit preserves both source records and does not infer links for
unmatched cards. It records 299,787 linked cards with a Hindi selection, 3,589
with an Urdu selection, 3,261 with both, 328 where Urdu fills an unmapped Hindi
card, and 154 with different mapped spellings. The release builder applies this
policy from `cross_script_edition_links.parquet` and its audit rather than adding
the two edition totals.

## Current national lookup and reproducibility

The 3.6 lookup contains 1,762,371 surnames and 2,596,042 retained surname-state
cells. It represents 736,861,903 retained source occurrences from 743,219,604
selected input occurrences. J&K contributes 1,942,676 retained occurrences,
Telangana 16,091,488 and Lakshadweep 3,311. National spelling consolidation uses
all state inputs, so replacing another state's source can change a state's
retained total even when its own input is unchanged.

The October 10 completion check rebuilt the J&K aggregate from all three
accepted handoffs and the edition links. The local national input directory
contained the older Andhra aggregate; rebuilding that table from its archived
Upnaam handoff restored the 3.6 input. The complete national rebuild reproduced
the published lookup and retained training table byte for byte. The runtime
lookup and training hashes remain unchanged.

The accepted inventories, source audits, three handoffs, romanization maps,
Urdu field provenance and edition links are retained under `data/jk_2018/`.
`make verify-jk` checks their SHA-256 digests against
`jk_2018_SHA256SUMS`. Person-level files remain untracked. Rebuild commands are
in [SOURCES.md](prep_er_data/SOURCES.md); use a new output directory.

The original recovery tree is archived at
`/Volumes/Staging/instate/jk_recovery_archive_20261010/`. Its
`ARCHIVE_MANIFEST.json` records each relative path, size and SHA-256. This includes
original PDFs, paid responses and cost records, source controls, historical
inventories, and experimental scripts. After rechecking every local and archived
file hash, the local recovery tree was removed. The completed full rolls and
enriched candidate are documented in [JK_2018_ROLLS.md](JK_2018_ROLLS.md).
Historical `data/jk_recovery/` paths in
audits refer to that archive. The retained local files support rerunning Upnaam
and rebuilding the national lookup without restoring the experimental tree.

The current evidence record is `jk_2018_completion.json`. Earlier national
candidate counts describe their historical source revisions, not the current
lookup. The retained 3.2 neural checkpoint and calibration remain in use because
the later retrained candidate performed worse on the frozen development
comparison. Lookup and neural fallback use different source revisions.

## Metered recovery record

Muse Spark 1.3 Contributor handled all paid OCR and transliteration review. The
OCR run completed 4,483 calls. One 26-card crop was abandoned after three
receipt-less timeouts and remains quarantined; 37 historical failed calls remain
archived. Provider-reported OCR usage was $3.13722599. Including reserved exposure
for receipt-less attempts and every earlier review phase, the conservative total
is $9.176521554, below the authorized $10 ceiling. No voter IDs, filenames,
locations, relationships or whole PDFs were sent for transliteration review.

## Source limits retained in the release

Four Urdu PDFs contain no structural rows: two are truncated, one contains only
a cover and map, and one contains control pages with a conflicting printed
identity. Another source has conflicting printed part identities. Unsupported
glyphs and fields remain missing; printed totals are never substituted for absent
records. The archive also contains 4,317 zero-filled Urdu files and 1,547 archived
AC/part keys without a readable PDF. These source gaps are explicit and do not
block publication of the recovered evidence.

Surname selection remains conservative. Corroboration can establish a repeatable
written-token rule, but it does not establish hereditary surname status or
accuracy for individual people. Confidence remains null. Raw elector rows, card
images and person-level handoffs remain restricted local artifacts.
