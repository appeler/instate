"""instate: State and language composition estimates for Indian surnames.

Public functions follow the appeler inference contract, composition form
(https://github.com/appeler/appellation): proportions across states or
languages that sum to one, explicit abstention instead of default
distributions, and shared metadata columns.

- ``lookup_state_composition``: a surname's processed occurrence shares
  across states in the 2017 Indian electoral rolls.
- ``estimate_state_composition``: the same quantity from a calibrated
  character-BiLSTM, for surnames outside the lookup table too.
- ``estimate_language_composition``: state composition mixed with Census
  2011 mother-tongue shares per state.
- ``lookup_state_official_languages`` and ``list_supported_states``:
  auxiliary reference lookups.

Outputs describe name patterns in stated reference populations. They do
not establish an individual's residence, origin, or language.
"""

from .composition import (
    estimate_language_composition,
    estimate_state_composition,
    lookup_state_composition,
)
from .reference import list_supported_states, lookup_state_official_languages

__all__ = [
    "estimate_language_composition",
    "estimate_state_composition",
    "list_supported_states",
    "lookup_state_composition",
    "lookup_state_official_languages",
]

try:
    from importlib.metadata import version

    __version__ = version("instate")
except ImportError:
    __version__ = "0+unknown"
