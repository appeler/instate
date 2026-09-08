"""Tests for the electoral-roll data builder."""

import csv
import gzip
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from model_training.prep_er_data.name_tables import (
    _build_remap,
    _load_word_map_nukta_tolerant,
    _resolve_last_name,
    build_last_names,
    name_counts2_english,
    name_counts_via_corpus,
    repair_devanagari_pdf,
    resolve_household,
    write_name_table,
    write_name_table2,
)

BENGALI = re.compile(r"[ঀ-৿]+")
NO_STOP = frozenset()


class TestNameTables(unittest.TestCase):
    def test_aggregate_romanize_and_drop_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            roll = Path(tmp) / "roll.csv"
            with open(roll, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["elector_name", "state"])
                w.writerow(["রাম দাস", "wb"])
                w.writerow(["রাম দাস", "wb"])
                w.writerow(["রাম", "wb"])
                w.writerow(["রাম খ", "wb"])  # খ absent from map -> residual -> dropped
            word_map = {"রাম": "ram", "দাস": "das"}
            counts, stats = name_counts_via_corpus(
                roll, name_col="elector_name", native_run=BENGALI, word_map=word_map
            )
            self.assertEqual(counts["ram das"], 2)
            self.assertEqual(counts["ram"], 1)
            self.assertEqual(stats["total_voters"], 4)
            self.assertEqual(stats["residual_voters"], 1)

    def test_write_name_table_sorted_by_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "names_wb.csv.gz"
            n = write_name_table(Counter({"ram": 5, "das": 9, "ali": 9}), out)
            self.assertEqual(n, 3)
            with gzip.open(out, "rt", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["english_name", "n_times"])
            self.assertEqual(rows[1:], [["ali", "9"], ["das", "9"], ["ram", "5"]])


class TestResolveLastName(unittest.TestCase):
    def test_t1_shared_token_position_independent(self):
        # shared non-honorific token wins even when it is first in the voter name
        self.assertEqual(
            _resolve_last_name("dawa sherpa", "passang sherpa", NO_STOP),
            ("sherpa", "T1"),
        )
        self.assertEqual(
            _resolve_last_name("sharma anil", "sharma ram", NO_STOP), ("sharma", "T1")
        )

    def test_t1_surname_first_convention(self):
        # Maharashtra "Surname Given FatherGiven": both patil & shankar are shared, but
        # the shared LEADING token (patil) is the surname, not the trailing one (shankar).
        self.assertEqual(
            _resolve_last_name("patil parvati shankar", "patil shankar", NO_STOP),
            ("patil", "T1"),
        )
        self.assertEqual(
            _resolve_last_name("jadhav sunita sanjay", "jadhav sanjay", NO_STOP),
            ("jadhav", "T1"),
        )
        # surname-last still works when leading tokens differ
        self.assertEqual(
            _resolve_last_name("sachin ramesh tendulkar", "ramesh tendulkar", NO_STOP),
            ("tendulkar", "T1"),
        )

    def test_particle_strip_enables_share(self):
        # "she" (Sheikh) is a particle -> stripped, so "husen" matches across both
        self.assertEqual(
            _resolve_last_name("she husen", "she husen", NO_STOP), ("husen", "T1")
        )
        # md prefix stripped; akram is then a single token -> inherit father
        self.assertEqual(
            _resolve_last_name("md akram", "md gulam rasul", NO_STOP),
            ("rasul", "T3"),
        )

    def test_t2_honorific_final_kept(self):
        # honorific suffixes are content by default (geographically predictive)
        self.assertEqual(
            _resolve_last_name("suresh kumar", "", NO_STOP), ("kumar", "T2")
        )
        self.assertEqual(_resolve_last_name("sunita devi", "", NO_STOP), ("devi", "T2"))

    def test_t3_single_token_voter_inherits_father(self):
        self.assertEqual(
            _resolve_last_name("purusha", "ramesh gupta", NO_STOP), ("gupta", "T3")
        )

    def test_t4_drop_single_token_no_father(self):
        self.assertEqual(_resolve_last_name("purusha", "", NO_STOP), (None, "DROP"))
        # all tokens too short / null
        self.assertEqual(_resolve_last_name("a b", "fnu", NO_STOP), (None, "DROP"))

    def test_singh_mode_stop(self):
        stop = frozenset({"singh", "kaur"})
        # default: singh is content -> selected as the surname
        self.assertEqual(_resolve_last_name("ram singh", "", NO_STOP), ("singh", "T2"))
        # stopped: singh can't be selected, so T2 falls back to the prior content token
        # ("ram"). (Known caveat of stopping a final honorific -- the default keeps them.)
        self.assertEqual(_resolve_last_name("ram singh", "", stop), ("ram", "T2"))
        # but a lone stopped token drops
        self.assertEqual(_resolve_last_name("singh", "kaur", stop), (None, "DROP"))


class TestBuildLastNames(unittest.TestCase):
    def test_end_to_end_weighting_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # write a tiny names_<slug> table
            src = tmp / "names_demo.csv.gz"
            write_name_table2(
                Counter(
                    {
                        ("dawa sherpa", "passang sherpa"): 5,  # T1 sherpa
                        ("suresh kumar", ""): 3,  # T2 kumar
                        ("purusha", "ramesh gupta"): 2,  # T3 gupta
                        ("solo", ""): 7,  # drop
                    }
                ),
                src,
            )
            st = build_last_names("demo", tmp, tmp)
            self.assertEqual(st["total"], 17)
            self.assertEqual(st["kept"], 10)  # 5 + 3 + 2
            self.assertEqual(st["tiers"]["DROP"], 7)
            with gzip.open(tmp / "last_names_demo.csv.gz", "rt") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["last_name", "n_times"])
            self.assertEqual(
                {r[0]: int(r[1]) for r in rows[1:]},
                {"sherpa": 5, "kumar": 3, "gupta": 2},
            )


class TestBuildRemap(unittest.TestCase):
    def test_deletion_only_artifact_gated(self):
        freqs = {
            "patil": 1000,
            "patila": 300,  # trailing-vowel artifact (Karnataka)
            "sah": 1000,
            "saha": 300,  # real Bengali surname, NOT artifact-state
            "ram": 5000,
            "rao": 300,  # substitution, not a deletion
        }
        art = {  # share of weight from ARTIFACT_STATES
            "patila": 0.9,
            "patil": 0.1,
            "saha": 0.05,
            "sah": 0.2,
            "rao": 0.9,
            "ram": 0.1,
        }
        anchors = {"patil", "sah", "ram"}
        remap = _build_remap(freqs, art, anchors, min_variant=100, art_min=0.6)
        self.assertEqual(
            remap.get("patila"), "patil"
        )  # deletion + artifact-state -> merge
        self.assertNotIn("saha", remap)  # not artifact-concentrated -> kept
        self.assertNotIn(
            "rao", remap
        )  # "ram" is a substitution, not a deletion -> kept


class TestDevanagariPdfRepair(unittest.TestCase):
    def test_artifacts_repaired(self):
        cases = {
            "अशोक कु मार": ["अशोक", "कुमार"],  # stub split off the word
            "राके श कु मार": ["राकेश", "कुमार"],  # lone consonant split off the end
            "राहुल शमार्": ["राहुल", "शर्मा"],  # reph printed after its syllable
            "िवजय िसंह": ["विजय", "सिंह"],  # i-matra printed before its consonant
            "गु�ता राम": ["राम"],  # conjunct lost to U+FFFD -> token dropped
            "ख़ान": ["खान"],  # nukta stripped for lookup
            "राम लाल": ["राम", "लाल"],
        }
        for raw, want in cases.items():
            self.assertEqual(repair_devanagari_pdf(raw), want, raw)

    def test_word_map_exact_before_nukta_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "hindi.csv.gz"
            with gzip.open(corpus, "wt", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["hindi", "english"])
                w.writerow(["क़ुमार", "qumar"])  # nukta variant listed first
                w.writerow(["कुमार", "kumar"])
                w.writerow(["ख़ान", "khan"])
            wm = _load_word_map_nukta_tolerant(corpus)
            self.assertEqual(wm["कुमार"], "kumar")
            self.assertEqual(wm["खान"], "khan")


class TestHouseholdTier(unittest.TestCase):
    def test_shared_token_wins_at_either_end(self):
        household = [
            ("etham jayamma", "balakishtaiah"),  # surname first
            ("etham ramulu", "balakishtaiah"),  # surname first
            ("kavita etham", "ramulu etham"),  # surname last, also shared with husband
            ("balakishtaiah", ""),  # single token, shares nothing -> T1..T3 fallback
        ]
        got = resolve_household(household, NO_STOP)
        self.assertEqual(got[:3], [("etham", "T0"), ("etham", "T0"), ("etham", "T0")])
        self.assertEqual(got[3], (None, "DROP"))

    def test_relation_token_breaks_ties_and_singletons_fall_through(self):
        # kumar is shared by two members; the father's name settles the surname
        household = [("anil kumar reddy", "suresh reddy"), ("sunil kumar reddy", "")]
        got = resolve_household(household, NO_STOP)
        self.assertEqual(got[0], ("reddy", "T0"))
        self.assertEqual(got[1][1], "T0")
        alone = resolve_household([("kavita namala", "anjaneyulu")], NO_STOP)
        self.assertEqual(alone, [("namala", "T2")])


class TestHouseholdSpellings(unittest.TestCase):
    def test_long_token_variants_merge_to_majority(self):
        household = [
            ("kiran kumar komatiareddy", "chandra komatireddy"),
            ("shravan kumar komatireddy", "chandra komatireddy"),
            ("andamma komatireddy", ""),
        ]
        got = resolve_household(household, NO_STOP)
        self.assertEqual({ln for ln, _ in got}, {"komatireddy"})
        self.assertEqual({tier for _, tier in got}, {"T0"})
        # nine letters, two edits apart: not the same spelling under the slack rule
        apart = resolve_household(
            [("narasimha thumkunta", ""), ("shiva tumukunta", "")], NO_STOP
        )
        self.assertEqual([ln for ln, _ in apart], ["thumkunta", "tumukunta"])

    def test_four_to_six_letters_merge_on_vowel_or_h_only(self):
        merge = [
            ("goud", "gaud"),
            ("begam", "begum"),
            ("sing", "singh"),
            ("jadav", "jadhav"),
        ]
        keep = [
            ("rani", "ravi"),
            ("rajesh", "ramesh"),
            ("kaleem", "saleem"),
            ("raju", "ramu"),
        ]
        for a, b in merge:
            got = resolve_household([(f"sita {a}", ""), (f"gita {b}", "")], NO_STOP)
            self.assertEqual(
                {ln for ln, _ in got}, {min(a, b, key=lambda t: (len(t), t))}, (a, b)
            )
        for a, b in keep:
            got = resolve_household([(f"sita {a}", ""), (f"gita {b}", "")], NO_STOP)
            self.assertEqual([ln for ln, _ in got], [a, b], (a, b))

    def test_short_tokens_never_fuzz(self):
        # ram / rao differ by one letter but short tokens must match exactly
        got = resolve_household(
            [("sita ram", "hari ram"), ("gita rao", "hari rao")], NO_STOP
        )
        self.assertEqual(got, [("ram", "T1"), ("rao", "T1")])


class TestRollSql(unittest.TestCase):
    def test_lookup_and_training_share_retained_cells_and_denominators(self):
        import duckdb
        from click.testing import CliRunner

        from model_training.prep_er_data.name_tables import cli

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for state, counts in {
                "lakshadweep": {"veda": 1, "mila": 2, "rama": 3, "aruna": 2},
                "kerala": {"veda": 5, "mila": 3, "rama": 5, "aruna": 1},
            }.items():
                write_name_table(
                    Counter(counts),
                    root / f"last_names_{state}.csv.gz",
                    header=("last_name", "n_times"),
                )
            for minimum in (3, 6):
                output = root / "lookup.parquet"
                training = root / "training.csv.gz"
                result = CliRunner().invoke(
                    cli,
                    [
                        "ln-prop",
                        "--in-dir",
                        str(root),
                        "--out",
                        str(output),
                        "--train-out",
                        str(training),
                        "--no-canon",
                        "--min-total",
                        str(minimum),
                    ],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                table = duckdb.read_parquet(str(output)).df().set_index("last_name")
                with gzip.open(training, "rt") as handle:
                    cells = {
                        (row["last_name"], row["state"]): int(row["n_times"])
                        for row in csv.DictReader(handle)
                    }
                observed = {}
                for name, row in table.iterrows():
                    released = 0
                    for state in ("Kerala", "Lakshadweep"):
                        count = round(row[state] * row.total_n)
                        if count:
                            self.assertGreaterEqual(count, 3)
                            observed[name, state] = count
                            released += count
                    self.assertEqual(released, row.total_n)
                self.assertEqual(observed, cells)
                self.assertNotIn("aruna", table.index)
                self.assertEqual(table.loc["rama", "total_n"], 8)
                if minimum == 3:
                    self.assertEqual(table.loc["veda", "total_n"], 5)
                    self.assertEqual(table.loc["veda", "Lakshadweep"], 0)

    def test_coalesce_and_where_on_split_relation_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            roll = Path(tmp) / "roll.csv"
            with open(roll, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "father's name", "husband's name", "section"])
                w.writerow(["Sona Patel", "", "Ramesh Patel", "main"])
                w.writerow(["Sona Patel", "Kanti Patel", "", "main"])
                w.writerow(["Sona Patel", "", "", "deleted"])
            counts, stats = name_counts2_english(
                roll,
                voter_col="name",
                father_col="father's name,husband's name",
                where="section = 'main'",
            )
            self.assertEqual(stats["total_voters"], 2)
            self.assertEqual(counts[("sona patel", "ramesh patel")], 1)
            self.assertEqual(counts[("sona patel", "kanti patel")], 1)

    def test_parquet_source(self):
        import duckdb

        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "roll.parquet"
            duckdb.connect().execute(
                f"COPY (SELECT 'Ram Das' AS elector_name, 'Hari Das' AS "
                f"father_or_husband_name) TO '{pq}' (FORMAT parquet)"
            )
            counts, _ = name_counts2_english(
                pq, voter_col="elector_name", father_col="father_or_husband_name"
            )
            self.assertEqual(counts[("ram das", "hari das")], 1)

    def test_upnaam_artifact_counts_resolved_surnames(self):
        import duckdb
        import gzip

        from model_training.prep_er_data.name_tables import build_last_names_upnaam

        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "roll's surnames.parquet"
            duckdb.connect().execute(
                f"COPY (SELECT * FROM (VALUES "
                "('kunninamel', 'house', false), ('kunninamel', 'household', false), "
                "(NULL, NULL, true)) t(surname_latin_normalized, surname_evidence, "
                "abstained)) TO ? (FORMAT parquet)",
                [str(pq)],
            )
            st = build_last_names_upnaam("lakshadweep", pq, Path(tmp))
            self.assertEqual((st["total"], st["kept"]), (3, 2))
            self.assertEqual(st["evidence"], {"house": 1, "household": 1})
            with gzip.open(st["out"], "rt") as fh:
                self.assertEqual(
                    fh.read().split(), ["last_name,n_times", "kunninamel,2"]
                )


if __name__ == "__main__":
    unittest.main()
