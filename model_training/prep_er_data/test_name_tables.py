"""Tests for the Phase 1 name-count build script (run in eroll's 3.13 venv).

Not collected by instate's default pytest (testpaths=["tests"]); run explicitly, e.g.
    ../../../eroll_transliteration/.venv/bin/python -m pytest test_name_tables.py
"""

import csv
import gzip
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from name_tables import (
    _build_remap,
    _resolve_last_name,
    build_last_names,
    name_counts_via_corpus,
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


if __name__ == "__main__":
    unittest.main()
