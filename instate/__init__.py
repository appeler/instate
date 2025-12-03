"""
instate: Predict state and language from Indian lastnames.

This package provides functions to:
1. Look up state distributions from 2017 Indian electoral rolls
2. Predict states and languages using neural networks

Main functions:
- get_state_distribution: Get P(state|lastname) from electoral rolls
- get_state_languages: Map states to official languages
- predict_state: Neural prediction of most likely states
- predict_language: Neural prediction of most likely languages
"""

from .electoral import (
    get_state_distribution,
    get_state_languages,
    list_available_states,
)
from .predict import predict_language, predict_state

__all__ = [
    "get_state_distribution",
    "get_state_languages",
    "predict_state",
    "predict_language",
    "list_available_states",
]

try:
    from importlib.metadata import version

    __version__ = version("instate")
except ImportError:
    __version__ = "1.0.0"
