"""Constants for the instate package."""

from __future__ import annotations

# Character to index mapping shared by every model input path.
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

VOCAB_SIZE = len(CHAR_TO_IDX)

# Electoral-roll data predates the union of these territories. Language data uses
# the current combined territory name.
STATE_LANGUAGE_ALIASES = {
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
}

# State labels
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
    # v2 (rebuilt from rolls) adds the three states the legacy v1 table omitted.
    "Himachal Pradesh",
    "Tamil Nadu",
    "West Bengal",
]

# State char-BiLSTM configuration (v1.2.0). Char vocab reuses CHAR_TO_IDX (27, <PAD>=0).
STATE_LSTM_EMB = 64
STATE_LSTM_HIDDEN = 384
STATE_LSTM_LAYERS = 2
STATE_LSTM_DROPOUT = 0.2
