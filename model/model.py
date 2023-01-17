import pandas as pd
from glob import glob
import os


def model(df):
    return df


def process_data(base_path):
    unprocessed_data = _read_gz_csv(base_path)
    import pdb

    pdb.set_trace()
    df = unprocessed_data
    return df


def _read_gz_csv(base_path):
    return pd.read_csv(base_path)


if __name__ == "__main__":
    chunk_size = 1000000
    base_dir = "/Users/dhingratul/Documents/instate_data"
    fid = "instate_parsed_0.csv.gz"
    df = process_data(os.path.join(base_dir, fid))
