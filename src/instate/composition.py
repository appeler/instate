"""State and language composition results for Indian surnames.

Three public functions share one result shape, the composition form of the
appeler inference contract: proportions that sum to one across states or
languages, explicit abstention, and the contract's common metadata columns.

- ``lookup_state_composition`` reports a surname's processed 2017
  electoral-roll occurrence shares across 34 states.
- ``estimate_state_composition`` runs the calibrated character-BiLSTM for
  the same quantity, including surnames outside the lookup table.
- ``estimate_language_composition`` mixes a surname's state composition
  with Census 2011 mother-tongue shares per state.

None of these estimates an individual's residence, origin, or language.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from ._contract import ResultProvenance, metadata_columns, preserve_reserved_columns
from .constants import CHAR_TO_IDX, GT_KEYS

_CACHE: dict[str, object] = {}

MINIMUM_MODEL_INPUT_LETTERS = 3

_STATE_SHARE_PREFIX = "state_share_"
_LANGUAGE_SHARE_PREFIX = "language_share_"
_RECORD_COUNT_COLUMN = "surname_record_count"
_BASIS_COLUMN = "language_basis"

_LOOKUP_REFERENCE_POPULATION = (
    "processed surname occurrences in included 2017 Indian electoral rolls"
)
_CENSUS_REFERENCE_POPULATION = (
    "Census of India 2011 C-16 mother-tongue populations by state"
)


def _slug(name: str) -> str:
    """Return a lowercase column-safe identifier for a state or language."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


STATE_SHARE_COLUMNS = {state: f"{_STATE_SHARE_PREFIX}{_slug(state)}" for state in GT_KEYS}


@dataclass(frozen=True)
class _ClassifiedName:
    """One input name after normalization."""

    canonical: str
    script_supported: bool | None
    reason: str | None


def _classify(value: object, *, for_model: bool) -> _ClassifiedName:
    """Map one raw input to a canonical surname or an abstention reason."""
    if not isinstance(value, str) or not value.strip():
        return _ClassifiedName("", None, "missing-name")
    stripped = value.strip()
    if not any(character.isalpha() for character in stripped):
        return _ClassifiedName("", None, "no-letters")
    canonical = "".join(
        character for character in stripped.lower() if character in CHAR_TO_IDX
    )
    if not canonical:
        return _ClassifiedName("", False, "unsupported-script")
    if for_model and len(canonical) < MINIMUM_MODEL_INPUT_LETTERS:
        return _ClassifiedName(canonical, True, "insufficient-evidence")
    return _ClassifiedName(canonical, True, None)


def _prepare(
    data: pd.DataFrame | pd.Series | list[str | None] | str,
    surname_column: str | None,
) -> tuple[pd.DataFrame, str]:
    """Validate input and return (frame, surname column name).

    Args:
        data: DataFrame, or surname sugar (string, list, Series).
        surname_column: Column holding surnames when ``data`` is a frame.

    Returns:
        The working frame and the name of its surname column.

    Raises:
        TypeError: If ``data`` is not a supported input type.
        ValueError: If the surname column is missing, absent, or duplicated.
    """
    if isinstance(data, pd.DataFrame):
        if surname_column is None:
            raise ValueError("surname_column is required for DataFrame input")
        if list(data.columns).count(surname_column) != 1:
            raise ValueError(
                f"surname column {surname_column!r} must appear exactly once"
            )
        return data, surname_column
    if isinstance(data, str):
        return pd.DataFrame({"surname": [data]}), "surname"
    if isinstance(data, pd.Series):
        return pd.DataFrame({"surname": data.to_numpy()}), "surname"
    if isinstance(data, list):
        return pd.DataFrame({"surname": data}), "surname"
    raise TypeError("data must be a DataFrame, Series, list, or string")


def _sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file, cached by path."""
    key = f"sha256:{path}"
    if key not in _CACHE:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        _CACHE[key] = digest.hexdigest()
    return str(_CACHE[key])


def _electoral_table() -> pd.DataFrame:
    """Load the electoral share table indexed by canonical surname."""
    if "electoral" not in _CACHE:
        path = Path(__file__).parent / "data" / "instate_unique_ln_state_prop_v2.parquet"
        table = pd.read_parquet(path)
        missing = set(GT_KEYS) - set(table.columns)
        if missing or "total_n" not in table.columns:
            raise RuntimeError("electoral table does not match the state vocabulary")
        _CACHE["electoral_revision"] = f"sha256:{_sha256(path)[:16]}"
        _CACHE["electoral"] = table.set_index("last_name")
    return _CACHE["electoral"]  # type: ignore[return-value]


def _language_shares() -> pd.DataFrame:
    """Load the census language-share matrix, states by languages."""
    if "language_shares" not in _CACHE:
        directory = Path(__file__).parent / "data"
        path = directory / "state_language_shares.parquet"
        manifest = json.loads(
            (directory / "state_language_shares.manifest.json").read_text("utf-8")
        )
        if _sha256(path) != manifest["artifact"]["sha256"]:
            raise RuntimeError("state_language_shares.parquet fails its manifest hash")
        table = pd.read_parquet(path)
        matrix = table.pivot(index="state", columns="language", values="share")
        if set(matrix.index) != set(GT_KEYS):
            raise RuntimeError("language shares do not cover the state vocabulary")
        ordered = sorted(column for column in matrix.columns if column != "other")
        matrix = matrix.loc[GT_KEYS, [*ordered, "other"]]
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise RuntimeError("language shares do not sum to one per state")
        _CACHE["language_shares_revision"] = f"sha256:{_sha256(path)[:16]}"
        _CACHE["language_shares"] = matrix
    return _CACHE["language_shares"]  # type: ignore[return-value]


def _calibrated_model() -> tuple[torch.nn.Module, float, str]:
    """Load the state model and its temperature; return (model, T, revision)."""
    if "state_model" not in _CACHE:
        from ._resources import resolve_model
        from .constants import (
            STATE_LSTM_DROPOUT,
            STATE_LSTM_EMB,
            STATE_LSTM_HIDDEN,
            STATE_LSTM_LAYERS,
            VOCAB_SIZE,
        )
        from .nnets import StateLSTM

        checkpoint = resolve_model("instate_state_lstm.pt")
        calibration_path = resolve_model("instate_state_lstm_calibration.json")
        calibration = json.loads(Path(calibration_path).read_text("utf-8"))
        temperature = float(calibration["temperature"])
        if temperature <= 0:
            raise RuntimeError("calibration temperature must be positive")
        model = StateLSTM(
            VOCAB_SIZE,
            len(GT_KEYS),
            STATE_LSTM_EMB,
            STATE_LSTM_HIDDEN,
            STATE_LSTM_LAYERS,
            STATE_LSTM_DROPOUT,
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.eval()
        revision = (
            f"sha256:{_sha256(checkpoint)[:16]}+{_sha256(calibration_path)[:16]}"
        )
        _CACHE["state_model"] = (model, temperature, revision)
    return _CACHE["state_model"]  # type: ignore[return-value]


def _model_state_matrix(canonicals: list[str]) -> np.ndarray:
    """Return calibrated state probabilities, one row per canonical surname."""
    from .nnets import pad_encoded

    model, temperature, _ = _calibrated_model()
    rows = np.full((len(canonicals), len(GT_KEYS)), np.nan)
    batch_size = 1024
    for start in range(0, len(canonicals), batch_size):
        chunk = canonicals[start : start + batch_size]
        encoded = [[CHAR_TO_IDX[c] for c in name] for name in chunk]
        x, lengths = pad_encoded(encoded)
        with torch.no_grad():
            logits = model(x, lengths) / temperature
            probabilities = torch.softmax(logits, dim=1)
        rows[start : start + len(chunk)] = probabilities.numpy()
    return rows


def _finish(
    data: pd.DataFrame,
    value_columns: dict[str, np.ndarray | pd.api.extensions.ExtensionArray],
    provenance: ResultProvenance,
    classified: list[_ClassifiedName],
    scored: list[bool],
) -> pd.DataFrame:
    """Assemble the result frame: inputs, value columns, contract block."""
    metadata = metadata_columns(
        provenance,
        scored=scored,
        script_supported=[item.script_supported for item in classified],
        abstention_reasons=[
            None if was_scored else item.reason
            for item, was_scored in zip(classified, scored, strict=True)
        ],
    )
    reserved = [*value_columns, *metadata]
    result = preserve_reserved_columns(data, reserved)
    for name, values in {**value_columns, **metadata}.items():
        result[name] = values
    return result


def lookup_state_composition(
    data: pd.DataFrame | pd.Series | list[str | None] | str,
    surname_column: str | None = None,
) -> pd.DataFrame:
    """Look up a surname's state composition in the 2017 electoral rolls.

    Each share is the fraction of the surname's included, processed roll
    occurrences recorded in a state. Unknown surnames abstain; they do not
    receive a default distribution.

    Args:
        data: DataFrame of inputs, or a surname string, list, or Series.
        surname_column: Column holding surnames for DataFrame input.

    Returns:
        A copy of the input with 34 ``state_share_*`` columns,
        ``surname_record_count``, and the contract metadata columns.
    """
    frame, column = _prepare(data, surname_column)
    classified = [_classify(value, for_model=False) for value in frame[column]]
    table = _electoral_table()

    shares = np.full((len(classified), len(GT_KEYS)), np.nan)
    counts: list[int | None] = [None] * len(classified)
    scored: list[bool] = [False] * len(classified)
    for row, item in enumerate(classified):
        if item.reason is not None:
            continue
        try:
            record = table.loc[item.canonical]
        except KeyError:
            classified[row] = _ClassifiedName(
                item.canonical, item.script_supported, "out-of-dictionary"
            )
            continue
        shares[row] = record[GT_KEYS].to_numpy(dtype=float)
        counts[row] = int(record["total_n"])
        scored[row] = True

    value_columns: dict[str, np.ndarray | pd.api.extensions.ExtensionArray] = {
        STATE_SHARE_COLUMNS[state]: pd.array(shares[:, index], dtype="Float64")
        for index, state in enumerate(GT_KEYS)
    }
    value_columns[_RECORD_COUNT_COLUMN] = pd.array(counts, dtype="Int64")

    _electoral_table()
    provenance = ResultProvenance(
        target="state-composition",
        input_scope="last-name",
        model_id="instate-electoral-state-lookup",
        model_version=_package_version(),
        model_revision=str(_CACHE["electoral_revision"]),
        reference_population=_LOOKUP_REFERENCE_POPULATION,
        calibration_status="not-applicable",
        calibration_reference="not-applicable",
    )
    return _finish(frame, value_columns, provenance, classified, scored)


def estimate_state_composition(
    data: pd.DataFrame | pd.Series | list[str | None] | str,
    surname_column: str | None = None,
) -> pd.DataFrame:
    """Estimate a surname's state composition with the calibrated model.

    The character-BiLSTM targets the same quantity the lookup reports and
    generalizes to surnames outside the lookup table. Probabilities are
    temperature-scaled on held-out surnames.

    Args:
        data: DataFrame of inputs, or a surname string, list, or Series.
        surname_column: Column holding surnames for DataFrame input.

    Returns:
        A copy of the input with 34 ``state_share_*`` columns and the
        contract metadata columns.
    """
    frame, column = _prepare(data, surname_column)
    classified = [_classify(value, for_model=True) for value in frame[column]]
    scored = [item.reason is None for item in classified]

    shares = np.full((len(classified), len(GT_KEYS)), np.nan)
    usable = [row for row, ok in enumerate(scored) if ok]
    if usable:
        matrix = _model_state_matrix([classified[row].canonical for row in usable])
        for position, row in enumerate(usable):
            shares[row] = matrix[position]

    value_columns: dict[str, np.ndarray | pd.api.extensions.ExtensionArray] = {
        STATE_SHARE_COLUMNS[state]: pd.array(shares[:, index], dtype="Float64")
        for index, state in enumerate(GT_KEYS)
    }
    _, _, revision = _calibrated_model()
    provenance = ResultProvenance(
        target="state-composition",
        input_scope="last-name",
        model_id="instate-state-lstm",
        model_version=_package_version(),
        model_revision=revision,
        reference_population=_LOOKUP_REFERENCE_POPULATION,
        calibration_status="temperature-scaled",
        calibration_reference=(
            "record-weighted validation surnames held out from training"
        ),
    )
    return _finish(frame, value_columns, provenance, classified, scored)


def estimate_language_composition(
    data: pd.DataFrame | pd.Series | list[str | None] | str,
    surname_column: str | None = None,
    *,
    basis: Literal["auto", "lookup", "model"] = "auto",
) -> pd.DataFrame:
    """Estimate a surname's language composition from state evidence.

    The language shares are defined, not observed: the surname's state
    composition mixed with each state's Census 2011 mother-tongue shares.
    The mixing assumes language and surname are independent within a state,
    which understates community-specific associations.

    Args:
        data: DataFrame of inputs, or a surname string, list, or Series.
        surname_column: Column holding surnames for DataFrame input.
        basis: State evidence to mix. ``lookup`` uses the electoral table
            and abstains on unknown surnames; ``model`` uses the calibrated
            LSTM; ``auto`` prefers the lookup and falls back to the model.

    Returns:
        A copy of the input with ``language_share_*`` columns, a
        ``language_basis`` column, and the contract metadata columns.

    Raises:
        ValueError: If ``basis`` is not one of the documented options.
    """
    if basis not in ("auto", "lookup", "model"):
        raise ValueError("basis must be 'auto', 'lookup', or 'model'")
    frame, column = _prepare(data, surname_column)
    for_model = basis != "lookup"
    classified = [_classify(value, for_model=for_model) for value in frame[column]]
    table = _electoral_table()
    languages = _language_shares()

    state_rows = np.full((len(classified), len(GT_KEYS)), np.nan)
    bases: list[str | None] = [None] * len(classified)
    model_rows: list[int] = []
    for row, item in enumerate(classified):
        if item.reason is not None:
            continue
        in_table = basis != "model" and item.canonical in table.index
        if in_table:
            state_rows[row] = table.loc[item.canonical, GT_KEYS].to_numpy(dtype=float)
            bases[row] = "electoral-lookup"
        elif basis == "lookup":
            classified[row] = _ClassifiedName(
                item.canonical, item.script_supported, "out-of-dictionary"
            )
        elif len(item.canonical) < MINIMUM_MODEL_INPUT_LETTERS:
            classified[row] = _ClassifiedName(
                item.canonical, item.script_supported, "insufficient-evidence"
            )
        else:
            model_rows.append(row)
    if model_rows:
        matrix = _model_state_matrix([classified[row].canonical for row in model_rows])
        for position, row in enumerate(model_rows):
            state_rows[row] = matrix[position]
            bases[row] = "state-model"

    scored = [base is not None for base in bases]
    language_matrix = state_rows @ languages.to_numpy()

    value_columns: dict[str, np.ndarray | pd.api.extensions.ExtensionArray] = {
        f"{_LANGUAGE_SHARE_PREFIX}{_slug(language)}": pd.array(
            language_matrix[:, index], dtype="Float64"
        )
        for index, language in enumerate(languages.columns)
    }
    value_columns[_BASIS_COLUMN] = pd.array(bases, dtype="string")

    revisions = [str(_CACHE["electoral_revision"]), str(_CACHE["language_shares_revision"])]
    calibration_status = "not-applicable"
    if any(base == "state-model" for base in bases):
        _, _, model_revision = _calibrated_model()
        revisions.insert(1, model_revision)
        calibration_status = "temperature-scaled state model for out-of-dictionary rows"
    provenance = ResultProvenance(
        target="language-composition",
        input_scope="last-name",
        model_id="instate-language-mixture",
        model_version=_package_version(),
        model_revision="+".join(revisions),
        reference_population=(
            f"{_LOOKUP_REFERENCE_POPULATION}, mixed with "
            f"{_CENSUS_REFERENCE_POPULATION}"
        ),
        calibration_status=calibration_status,
        calibration_reference="see state-composition basis",
    )
    return _finish(frame, value_columns, provenance, classified, scored)


def _package_version() -> str:
    """Return the installed package version."""
    from . import __version__

    return str(__version__)
