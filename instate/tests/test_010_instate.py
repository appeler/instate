#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for in_rolls_fn.py

"""

import unittest
import pandas as pd
from instate.instate import lookup_lang

class TestInRollsLn(unittest.TestCase):
    def setUp(self):
        names = [{"name": "sood"}, {"name": "chintalapati"}]
        self.pred_lang = ["hindi", "telugu"]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_in_rolls_fn(self):
        odf = lookup_lang(self.df, "name")
        print(odf)
        self.assertIn("name", odf.columns)
        self.assertIn("predicted_lang", odf.columns)
        # check predicted_lang matches with pred_lang
        self.assertListEqual(odf["predicted_lang"].tolist(), self.pred_lang)
