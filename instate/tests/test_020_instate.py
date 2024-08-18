#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for in_rolls_fn.py

"""

import unittest
import pandas as pd
from instate import predict_lang

class TestInRollsLn(unittest.TestCase):
    def setUp(self):
        names = [{"name": "sood"}, {"name": "chintalapati"}]
        self.pred_lang = ["hindi", "telugu"]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_in_rolls_fn(self):
        odf = predict_lang(self.df, "name")
        print(odf)
        self.assertIn("name", odf.columns)
        self.assertIn("predicted_lang", odf.columns)
        self.assertListEqual(odf["predicted_lang"].apply(lambda x: x[0]).tolist(), self.pred_lang)

