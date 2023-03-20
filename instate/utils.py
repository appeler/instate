# -*- coding: utf-8 -*-

import sys
import os
from os import path
import pandas as pd
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from .nnets import infer, GRU_net, GT_KEYS, n_letters, n_hidden

def column_exists(df: pd.DataFrame, col: str) -> bool:
    """Check the column name exists in the DataFrame.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame.
        col (str): Column name.

    Returns:
        bool: True if exists, False if does not exist.

    """
    if col and (col not in df.columns):
        print(f"The specified column `{col}` was not found in the input file")
        return False
    else:
        return True

def find_ngrams(vocab: list, text: str, n: int) -> list:
    """Find and return list of the index of n-grams in the vocabulary list.

    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.

    Args:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int): N-grams

    Returns:
        list: List of the index of n-grams in the vocabulary list.

    """

    wi = []

    if not isstring(text):
        return wi

    a = zip(*[text[i:] for i in range(n)])
    for i in a:
        w = ''.join(i)
        try:
            idx = vocab.index(w)
        except Exception as e:
            idx = 0
        wi.append(idx)
    return wi


def get_app_file_path(app_name, filename):
    user_dir = path.expanduser('~')
    app_data_dir = path.join(user_dir, '.' + app_name)
    if not path.exists(app_data_dir):
        os.makedirs(app_data_dir)
    file_path = path.join(app_data_dir, filename)
    return file_path


def download_file(url, target) -> bool:

    headers = {}

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream = True, headers = headers)

    if r.status_code == 200:
        chunk_size = (64 * 1024)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0)) / chunk_size

        total_size += 1

        with open(target, 'wb') as f:
            for data in tqdm(r.iter_content(chunk_size), total=round(total_size, 1), unit_scale=chunk_size/1024, unit='KB'):
                f.write(data)
        return True
    else:
        print(f"ERROR: status_code={r.status_code}")
        return False

def _pred_last_state(model, name: str, k: int=3):
    output = infer(model, name)
    _, indices = output.topk(k)
    idx_list = indices.numpy().flatten().tolist()
    preds = [GT_KEYS[i] for i in idx_list]
    return preds

def _load_model(path):
    device = torch.device('cpu')
    model = GRU_net(n_letters, n_hidden, len(GT_KEYS))
    model.load_state_dict(torch.load(path, map_location = device))
    return model
