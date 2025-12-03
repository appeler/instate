"""
Internal utilities for instate package.

This module contains helper functions for data loading, name cleaning, and caching.
These are internal utilities and not part of the public API.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import torch
import torch.nn as nn
from tqdm import tqdm

# Cache for loaded data
_CACHE = {}

# URLs for downloading data
ELECTORAL_DATA_URLS = {
    "v1": "https://github.com/appeler/instate/raw/main/data/instate_unique_ln_state_prop_v1.csv.gz",
}

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
        possible_cols = [
            c
            for c in df.columns
            if any(n in c.lower() for n in ["name", "lastname", "surname"])
        ]
        if not possible_cols:
            # Just use first column
            name_column = df.columns[0]
        else:
            name_column = possible_cols[0]

    # Ensure the name column is first
    if name_column != df.columns[0]:
        cols = [name_column] + [c for c in df.columns if c != name_column]
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
    if not isinstance(name, str):
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
    result["__cleaned_name"] = result[name_column].apply(clean_name)

    # Filter out invalid names
    result = result[result["__cleaned_name"].str.len() > 2]

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
    user_dir = os.path.expanduser("~")
    app_data_dir = os.path.join(user_dir, ".instate")

    if not os.path.exists(app_data_dir):
        os.makedirs(app_data_dir)

    return os.path.join(app_data_dir, filename)


def download_file(url: str, target: str) -> bool:
    """Download file with progress bar.

    Args:
        url: URL to download from
        target: Target file path

    Returns:
        True if successful, False otherwise
    """
    try:
        r = requests.get(url, stream=True)

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


def load_electoral_data(dataset: str = "v1") -> pd.DataFrame:
    """Load electoral rolls data, downloading if needed.

    Args:
        dataset: Dataset version to load

    Returns:
        DataFrame with electoral rolls data
    """
    global _CACHE

    cache_key = f"electoral_{dataset}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # Check if data exists locally
    filename = f"instate_unique_ln_state_prop_{dataset}.csv.gz"
    data_path = get_app_file_path(filename)

    if not os.path.exists(data_path):
        # First try to find the file in the package data directory
        package_data_dir = os.path.join(os.path.dirname(__file__), "data")
        local_path = os.path.join(package_data_dir, filename)

        if os.path.exists(local_path):
            print("Copying electoral rolls data from package...")
            import shutil

            shutil.copy2(local_path, data_path)
        else:
            print(f"Downloading electoral rolls data ({dataset})...")
            if not download_file(ELECTORAL_DATA_URLS[dataset], data_path):
                raise RuntimeError("Failed to download electoral data")

    # Load data
    df = pd.read_csv(data_path)
    df.rename(columns={"last_name": "__last_name"}, inplace=True)

    # Cache it
    _CACHE[cache_key] = df

    return df


def load_gru_model():
    """Load GRU model for state prediction.

    Returns:
        Loaded PyTorch model
    """
    global _CACHE

    if "gru_model" in _CACHE:
        return _CACHE["gru_model"]

    # Check if model exists
    model_path = get_app_file_path("instate_gru.pth")

    if not os.path.exists(model_path):
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


def load_lstm_model():
    """Load LSTM model for language prediction.

    Returns:
        Loaded PyTorch model and supporting data
    """
    global _CACHE

    if "lstm_model" in _CACHE:
        return _CACHE["lstm_model"], _CACHE["lstm_data"]

    # Import constants instead of loading from files
    from .constants import (
        CHAR_TO_IDX,
        IDX_TO_LANG,
        NUM_LANGUAGES,
        VOCAB_SIZE,
    )
    # LanguagePredictor is now defined in this file

    # Model configuration
    embedding_dim = 50
    hidden_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguagePredictor(VOCAB_SIZE, embedding_dim, hidden_dim, NUM_LANGUAGES)
    model.to(device)

    # Load weights
    data_dir = os.path.dirname(__file__)
    model_file = os.path.join(data_dir, "data", "state_lang_labels.pt")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))

    model.eval()

    # Cache everything
    lstm_data = {"char2idx": CHAR_TO_IDX, "idx2lang": IDX_TO_LANG, "device": device}

    _CACHE["lstm_model"] = model
    _CACHE["lstm_data"] = lstm_data

    return model, lstm_data


def load_language_lookup_data():
    """Load data for KNN language lookup.

    Returns:
        DataFrame with lastname to language mapping
    """
    import tarfile

    global _CACHE

    if "lang_lookup" in _CACHE:
        return _CACHE["lang_lookup"]

    data_file_name = "lastname_langs_india"
    data_path = get_app_file_path(data_file_name)

    if not os.path.exists(data_path + ".csv"):
        data_dir = os.path.dirname(__file__)
        gz_path = os.path.join(data_dir, "data", f"{data_file_name}.csv.tar.gz")
        print("Extracting language lookup data...")
        with tarfile.open(gz_path, "r:gz") as tar:
            tar.extract(f"{data_file_name}.csv", data_path, filter="data")

    df = pd.read_csv(f"{data_path}/{data_file_name}.csv")
    _CACHE["lang_lookup"] = df

    return df


class LanguagePredictor(nn.Module):
    """LSTM model for predicting languages from names.

    This model uses character embeddings and LSTM to predict the top 3 most
    likely languages for a given name.
    """

    def __init__(
        self, num_chars, embedding_dim=64, lstm_hidden_dim=128, num_languages=37
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, num_languages)
        self.fc2 = nn.Linear(lstm_hidden_dim, num_languages)
        self.fc3 = nn.Linear(lstm_hidden_dim, num_languages)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        out1 = self.fc1(h_n)
        out2 = self.fc2(h_n)
        out3 = self.fc3(h_n)
        return out1, out2, out3
