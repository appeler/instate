#!/usr/bin/env python

"""
Tests for the new instate API.

Tests all main functions with the new clean API.
"""

import unittest
import pandas as pd
import instate


class TestInstateAPI(unittest.TestCase):
    """Test the new instate API functions."""

    def setUp(self):
        """Set up test data."""
        self.names = ["sood", "chintalapati", "sharma"]
        self.states = ["Delhi", "Punjab", "Karnataka"]
        self.df_names = pd.DataFrame({"lastname": self.names})
        self.df_states = pd.DataFrame({"state": self.states})

    def test_get_state_distribution_list(self):
        """Test electoral rolls lookup with list input."""
        result = instate.get_state_distribution(self.names)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("name", result.columns)
        # Should have 31 state columns + 1 name column = 32+ columns
        self.assertGreater(len(result.columns), 31)

    def test_get_state_distribution_dataframe(self):
        """Test electoral rolls lookup with DataFrame input."""
        result = instate.get_state_distribution(self.df_names, "lastname")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("lastname", result.columns)
        self.assertGreater(len(result.columns), 31)

    def test_predict_state(self):
        """Test GRU state prediction."""
        result = instate.predict_state(self.names, top_k=3)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("predicted_states", result.columns)
        
        # Check that we get lists of states
        predictions = result["predicted_states"].iloc[0]
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 3)

    def test_predict_language_lstm(self):
        """Test LSTM language prediction."""
        result = instate.predict_language(self.names, model="lstm", top_k=3)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("predicted_languages", result.columns)
        
        # Check that we get lists of languages
        predictions = result["predicted_languages"].iloc[0]
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 3)

    def test_predict_language_knn(self):
        """Test KNN language prediction."""
        result = instate.predict_language(self.names, model="knn")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("predicted_languages", result.columns)
        
        # KNN returns single best language
        prediction = result["predicted_languages"].iloc[0]
        self.assertIsInstance(prediction, str)

    def test_get_state_languages(self):
        """Test state to language mapping."""
        result = instate.get_state_languages(self.states)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("state", result.columns)
        self.assertIn("official_languages", result.columns)

    def test_list_available_states(self):
        """Test listing available states."""
        states = instate.list_available_states()
        
        self.assertIsInstance(states, list)
        self.assertGreater(len(states), 30)  # Should have 31 states
        self.assertIn("Delhi", states)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid model for state prediction
        with self.assertRaises(ValueError):
            instate.predict_state(self.names, model="invalid")
        
        # Invalid model for language prediction
        with self.assertRaises(ValueError):
            instate.predict_language(self.names, model="invalid")

    def test_empty_input(self):
        """Test handling of empty inputs."""
        result = instate.get_state_distribution([])
        self.assertEqual(len(result), 0)
        
        result = instate.predict_state([])
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()