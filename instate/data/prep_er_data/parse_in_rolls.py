import pandas as pd
from glob import glob
import os
import re
import numpy as np


def select_csv_folder(base_dir):
    all_data = []
    for fn in sorted(glob(os.path.join(base_dir, "*.csv"))):
        state_split = re.split("[- _ : + .]", os.path.basename(fn))
        state = state_split[0]
        print(f"Processing state: {state}")
        if "guj" in state_split[1:]:
            print(f"Skipping non-supported state: {state}")
        else:
            print(f"Processing, csv folder,  state {state}")
            columns = pd.read_csv(fn, index_col=0, nrows=0).columns.tolist()
            if "name" in columns:
                df = pd.read_csv(
                    fn,
                    usecols=["name", "state", "father_or_husband_name", "sex"],
                )
                df = df.rename(columns={"name": "elector_name"})
            else:
                df = pd.read_csv(
                    fn,
                    usecols=["elector_name", "state",
                             "father_or_husband_name", "sex"],
                )
            df["state"] = state
            all_data.append(df)
    return pd.concat(all_data)


def select_gz_chunk(df, state_split):
    all_data = []
    if "clean" in state_split:
        df = df[["elector_name_t13n", "state",
                 "father_or_husband_name_t13n", "sex"]]
        df = df.rename(
            columns={
                "elector_name_t13n": "elector_name",
                "father_or_husband_name_t13n": "father_or_husband_name",
            }
        )
    else:
        df = df[["elector_name", "state", "father_or_husband_name", "sex"]]
    df["state"] = state_split[0]
    all_data.append(df)
    return pd.concat(all_data)


def _establish_last_name(name, father_name):
    if name is np.nan:
        name = "FNU"
    if father_name is np.nan:
        father_name = "FNU"
    if len(name.split()) > 1:
        last_name = name.split()[-1]
    else:
        if len(father_name.split()) > 1:
            last_name = father_name.split()[-1]
        else:
            last_name = "LNU"
    return last_name


def establish_last_name(df):
    print("Cleaning Data, processing last names")
    df["last_name"] = df.apply(
        lambda x: _establish_last_name(x["elector_name"], x["father_or_husband_name"]),
        axis=1,
    )
    df = df.drop(["father_or_husband_name", "elector_name"], axis=1)
    return df


def export_csv_gz(df, write_dir):
    path_to_write = os.path.join(write_dir, f"instate_selected.csv.gz")
    print(f"Writing dataframe to path: {path_to_write}")
    df.to_csv(path_to_write, compression="gzip", index=False)


if __name__ == "__main__":
    selected_data = []
    base_dir = "/Users/dhingratul/Documents/parsed"
    write_dir = "/Users/dhingratul/Documents/instate_data"
    chunk_size = 1000000
    unsupported_states = ["himachal", "wb", "tn"]

    # # "*csv"
    df_csv = select_csv_folder(base_dir)
    df_csv = establish_last_name(df_csv)
    selected_data.append(df_csv)

    # *.7z, pre-req: Extract .7z file to a folder manually using system unzipper
    for f in [
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]:
        folder_split = re.split("[- _ : +]", os.path.basename(f))
        state = folder_split[0]
        if "guj" in folder_split[1:]:
            print(f"Skipping unsupported state: {state}")
        else:
            print(f"Processing, 7z,  {state}")
            df_7z = select_csv_folder(os.path.join(base_dir, f))
            df_7z = establish_last_name(df_7z)
            selected_data.append(df_7z)

    # *.gz.csv, pre-req: use scripts/concatenate.py to merge .partaa, .partab, etc files
    for base_path in sorted(glob(os.path.join(base_dir, "*.csv.gz"))):
        state_split = re.split("[- _ : +]", os.path.basename(base_path))
        state = state_split[0]
        if state in unsupported_states:
            print(f"Skipping unsupported state: {state}")
        else:
            df_test = pd.read_csv(base_path, index_col=0, nrows=1)
            columns = df_test.columns.tolist()
            for df in pd.read_csv(base_path, chunksize=chunk_size):
                print(f"Processing, gz folder,  state: {state}, chunk {chunk_size}")
                df_gz = select_gz_chunk(df, state_split)
                df_gz = establish_last_name(df_gz)
                selected_data.append(df_gz)
    final_df = pd.concat(selected_data)
    final_df = final_df[final_df.last_name.str.isalpha()]
    final_df["last_name"] = final_df["last_name"].str.lower()
    final_df =  final_df[final_df["last_name"].str.contains('[a-z]',  na=False)]
    final_df = final_df[final_df.last_name.str.len() > 2]
    export_csv_gz(final_df, write_dir)
