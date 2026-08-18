"""instate: Rank state and synthetic language targets from Indian surnames.

This package provides functions to:
1. Look up state distributions from 2017 Indian electoral rolls
2. Predict states and languages using neural networks

Main functions:
- get_state_distribution: Get processed surname-occurrence state shares
- get_state_languages: Map states to official languages
- predict_state: Neural prediction of most likely states
- predict_language: Neural prediction of most likely languages
"""

from .electoral import (
    get_state_distribution,
    get_state_languages,
    list_available_states,
)
from .predict import get_model_metadata, predict_language, predict_state

__all__ = [
    "get_model_metadata",
    "get_state_distribution",
    "get_state_languages",
    "list_available_states",
    "predict_language",
    "predict_state",
]

try:
    from importlib.metadata import version

    __version__ = version("instate")
except ImportError:
    __version__ = "0+unknown"
