"""
Constants for the instate package.

Contains static data that was previously stored in small files.
"""

from __future__ import annotations

# Language list (originally from langs.txt)
LANGUAGES = [
    "sindhi",
    "nepali",
    "kannada",
    "marathi",
    "mizo",
    "adi",
    "garo",
    "tagin",
    "assamese",
    "hindi",
    "odia",
    "french",
    "punjabi",
    "naga languages",
    "english",
    "chenchu",
    "urdu",
    "bengali",
    "maithili",
    "dogri",
    "kokborok",
    "santali",
    "kashmiri",
    "gujarati",
    "apatani",
    "tulu",
    "konkani",
    "telugu",
    "malayalam",
    "tamil",
    "meitei",
    "khasi",
    "gondi",
    "bodo",
    "nishi",
    "chakma",
    "pahari and kumauni",
]

# Character to index mapping (originally from char2idx.json)
CHAR_TO_IDX = {
    "<PAD>": 0,
    "n": 1,
    "g": 2,
    "i": 3,
    "m": 4,
    "c": 5,
    "w": 6,
    "u": 7,
    "e": 8,
    "v": 9,
    "d": 10,
    "a": 11,
    "l": 12,
    "t": 13,
    "s": 14,
    "q": 15,
    "b": 16,
    "f": 17,
    "o": 18,
    "z": 19,
    "p": 20,
    "r": 21,
    "k": 22,
    "h": 23,
    "y": 24,
    "x": 25,
    "j": 26,
}

# Derived mappings for convenience
LANG_TO_IDX = {lang: idx for idx, lang in enumerate(LANGUAGES)}
IDX_TO_LANG = {idx: lang for lang, idx in LANG_TO_IDX.items()}

# Model dimensions
VOCAB_SIZE = len(CHAR_TO_IDX)
NUM_LANGUAGES = len(LANGUAGES)

# GRU model constants for state prediction
GT_KEYS = [
    "Andaman and Nicobar Islands",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Jharkhand",
    "Jammu and Kashmir and Ladakh",
    "Karnataka",
    "Kerala",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Madhya Pradesh",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
]

# GRU model configuration
GRU_HIDDEN_SIZE = 2048
GRU_ALL_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;"
GRU_N_LETTERS = len(GRU_ALL_LETTERS)
