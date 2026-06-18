"""
Internal utilities for instate package.

This module contains helper functions for data loading, name cleaning, and caching.
These are internal utilities and not part of the public API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import torch
from tqdm import tqdm

# Cache for loaded data
_CACHE: dict[str, object] = {}

# URLs for downloading data (fallback only). v2 ships bundled under instate/data/; v1 was
# dropped from the wheel in 1.2.0 but stays committed (model_training/data) + downloadable.
ELECTORAL_DATA_URLS = {
    "v1": "https://github.com/appeler/instate/raw/main/model_training/data/instate_unique_ln_state_prop_v1.csv.gz",
    "v2": "https://github.com/appeler/instate/raw/main/instate/data/instate_unique_ln_state_prop_v2.csv.gz",
}

# Legacy GRU weights (retired in 1.2.0; the state model is now the bundled BiLSTM).
MODEL_URLS = {
    "gru": "https://dataverse.harvard.edu/api/v1/access/datafile/6981460",
}


def prepare_name_dataframe(
    names: pd.DataFrame | list[str], name_column: str | None = None
) -> pd.DataFrame:
    """Convert input to DataFrame with standardized name column.

    Args:
        names: DataFrame or list of names
        name_column: Column name if DataFrame provided

    Returns:
        DataFrame with names in first column
    """
    if isinstance(names, list):
        return pd.DataFrame({"name": names})

    df = names.copy()

    # If no column specified, try to find one
    if name_column is None:
        # Look for common name columns
        possible_cols: list[str] = [
            str(c)
            for c in df.columns
            if any(n in str(c).lower() for n in ["name", "lastname", "surname"])
        ]
        if not possible_cols:
            # Just use first column
            name_column = str(df.columns[0])
        else:
            name_column = possible_cols[0]

    # Ensure the name column is first
    if name_column != str(df.columns[0]):
        cols = [name_column] + [str(c) for c in df.columns if str(c) != name_column]
        df = df[cols]

    return df


def clean_name(name: str) -> str:
    """Clean and standardize a single name.

    - Convert to lowercase
    - Strip whitespace
    - Remove non-alphabetic characters

    Args:
        name: Input name string

    Returns:
        Cleaned name
    """
    if not name:
        return ""

    # Basic cleaning
    cleaned = name.strip().lower()

    # Keep only alphabetic characters
    cleaned = "".join(c for c in cleaned if c.isalpha())

    return cleaned


def clean_names_in_df(df: pd.DataFrame, name_column: str) -> pd.DataFrame:
    """Clean names in a DataFrame column.

    Args:
        df: Input DataFrame
        name_column: Column containing names

    Returns:
        DataFrame with added __cleaned_name column and filtered rows
    """
    result = df.copy()

    # Handle empty DataFrame
    if len(result) == 0:
        result["__cleaned_name"] = pd.Series([], dtype=str)
        return result

    # Clean names
    result["__cleaned_name"] = result[name_column].apply(clean_name)  # type: ignore[reportUnknownMemberType]

    # Filter out invalid names
    result = result[result["__cleaned_name"].str.len() > 2]  # type: ignore[reportUnknownMemberType]

    # Drop duplicates based on cleaned name
    result = result.drop_duplicates(subset=["__cleaned_name"], keep="first")

    return result


def get_app_file_path(filename: str) -> str:
    """Get path for cached application data.

    Args:
        filename: Name of file

    Returns:
        Full path to file in app data directory
    """
    app_data_dir = Path.home() / ".instate"

    if not app_data_dir.exists():
        app_data_dir.mkdir(parents=True, exist_ok=True)

    return str(app_data_dir / filename)


def download_file(url: str, target: str) -> bool:
    """Download file with progress bar.

    Args:
        url: URL to download from
        target: Target file path

    Returns:
        True if successful, False otherwise
    """
    try:
        r = requests.get(url, stream=True, timeout=30)

        if r.status_code == 200:
            chunk_size = 64 * 1024
            total_size = int(r.headers.get("content-length", 0)) / chunk_size + 1

            with open(target, "wb") as f:
                for data in tqdm(
                    r.iter_content(chunk_size),
                    total=round(total_size, 1),
                    unit_scale=chunk_size / 1024,
                    unit="KB",
                    desc="Downloading",
                ):
                    f.write(data)
            return True
        else:
            print(f"Download failed with status code: {r.status_code}")
            return False
    except Exception as e:
        print(f"Download error: {e}")
        return False


def load_electoral_data(dataset: str = "v2") -> pd.DataFrame:
    """Load electoral rolls data, downloading if needed.

    Args:
        dataset: Dataset version to load. ``"v2"`` (default) covers all 34 states/UTs
            (rebuilt from the rolls); ``"v1"`` is the legacy 31-state table.

    Returns:
        DataFrame with electoral rolls data
    """
    global _CACHE

    cache_key = f"electoral_{dataset}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]  # type: ignore[return-value]

    # Check if data exists locally
    filename = f"instate_unique_ln_state_prop_{dataset}.csv.gz"
    data_path = get_app_file_path(filename)

    if not Path(data_path).exists():
        # First try to find the file in the package data directory
        package_data_dir = Path(__file__).parent / "data"
        local_path = package_data_dir / filename

        if local_path.exists():
            print("Copying electoral rolls data from package...")
            import shutil

            shutil.copy2(str(local_path), data_path)
        else:
            print(f"Downloading electoral rolls data ({dataset})...")
            if not download_file(ELECTORAL_DATA_URLS[dataset], data_path):
                raise RuntimeError("Failed to download electoral data")

    # Load data
    df = pd.read_csv(data_path)  # type: ignore[misc]
    df.rename(columns={"last_name": "__last_name"}, inplace=True)

    # Cache it
    _CACHE[cache_key] = df

    return df


def load_gru_model() -> torch.nn.Module:
    """Load GRU model for state prediction.

    Returns:
        Loaded PyTorch model
    """
    global _CACHE

    if "gru_model" in _CACHE:
        return _CACHE["gru_model"]  # type: ignore[return-value]

    # Check if model exists
    model_path = get_app_file_path("instate_gru.pth")

    if not Path(model_path).exists():
        print("Downloading GRU model...")
        if not download_file(MODEL_URLS["gru"], model_path):
            raise RuntimeError("Failed to download GRU model")

    # Load model
    from .constants import GRU_HIDDEN_SIZE, GRU_N_LETTERS, GT_KEYS
    from .nnets import GRU_net

    device = torch.device("cpu")
    model = GRU_net(GRU_N_LETTERS, GRU_HIDDEN_SIZE, len(GT_KEYS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _CACHE["gru_model"] = model

    return model


def load_state_lstm_model() -> torch.nn.Module:
    """Load the bundled char-BiLSTM state model (v1.2.0; no download).

    Returns:
        Loaded PyTorch model in eval mode.
    """
    global _CACHE

    if "state_lstm_model" in _CACHE:
        return _CACHE["state_lstm_model"]  # type: ignore[return-value]

    from .constants import (
        GT_KEYS,
        STATE_LSTM_DROPOUT,
        STATE_LSTM_EMB,
        STATE_LSTM_HIDDEN,
        STATE_LSTM_LAYERS,
        VOCAB_SIZE,
    )
    from .nnets import StateLSTM

    model_file = Path(__file__).parent / "data" / "instate_state_lstm.pt"
    model = StateLSTM(
        VOCAB_SIZE,
        len(GT_KEYS),
        STATE_LSTM_EMB,
        STATE_LSTM_HIDDEN,
        STATE_LSTM_LAYERS,
        STATE_LSTM_DROPOUT,
    )
    model.load_state_dict(
        torch.load(str(model_file), map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()

    _CACHE["state_lstm_model"] = model
    return model


def load_language_lstm_model() -> torch.nn.Module:
    """Load the bundled char-BiLSTM language model (v1.2.0; no download).

    Returns:
        Loaded PyTorch model in eval mode.
    """
    global _CACHE

    if "language_lstm_model" in _CACHE:
        return _CACHE["language_lstm_model"]  # type: ignore[return-value]

    from .constants import (
        LANG_LSTM_DROPOUT,
        LANG_LSTM_EMB,
        LANG_LSTM_HIDDEN,
        LANG_LSTM_LAYERS,
        NUM_LANGUAGES,
        VOCAB_SIZE,
    )
    from .nnets import CharBiLSTM

    model_file = Path(__file__).parent / "data" / "instate_lang_lstm.pt"
    model = CharBiLSTM(
        VOCAB_SIZE,
        NUM_LANGUAGES,
        LANG_LSTM_EMB,
        LANG_LSTM_HIDDEN,
        LANG_LSTM_LAYERS,
        LANG_LSTM_DROPOUT,
    )
    model.load_state_dict(
        torch.load(str(model_file), map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()

    _CACHE["language_lstm_model"] = model
    return model


def load_language_lookup_data() -> pd.DataFrame:
    """Load data for KNN language lookup.

    Returns:
        DataFrame with lastname to language mapping
    """
    import tarfile

    global _CACHE

    if "lang_lookup" in _CACHE:
        return _CACHE["lang_lookup"]  # type: ignore[return-value]

    data_file_name = "lastname_langs_india"
    data_path = get_app_file_path(data_file_name)

    csv_file_path = Path(data_path) / f"{data_file_name}.csv"
    if not csv_file_path.exists():
        gz_path = Path(__file__).parent / "data" / f"{data_file_name}.csv.tar.gz"
        print("Extracting language lookup data...")
        with tarfile.open(str(gz_path), "r:gz") as tar:
            tar.extract(f"{data_file_name}.csv", data_path, filter="data")

    df = pd.read_csv(str(csv_file_path))  # type: ignore[misc]
    _CACHE["lang_lookup"] = df

    return df
