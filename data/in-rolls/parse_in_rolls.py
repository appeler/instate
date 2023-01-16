import pandas as pd
from glob import glob
import os
import re
import numpy as np


def scrape_csv_folder(base_dir):
    all_data = None
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
                    usecols=["name", "state", "father_or_husband_name"],
                )
                df = df.rename(columns={"name": "elector_name"})
            else:
                df = pd.read_csv(
                    fn,
                    usecols=["elector_name", "state", "father_or_husband_name"],
                )
            df["state"] = state
            if all_data is None:
                all_data = df
            else:
                all_data = pd.concat([all_data, df])
    return all_data


def scrape_gz_chunk(df, state_split):
    all_data = None
    if "clean" in state_split:
        df_t13 = df[["elector_name_t13n", "state", "father_or_husband_name_t13n"]]
        df_t13 = df_t13.rename(
            columns={
                "elector_name_t13n": "elector_name",
                "father_or_husband_name_t13n": "father_or_husband_name",
            }
        )
    else:
        df_t13 = df[["elector_name", "state", "father_or_husband_name"]]
    df_t13["state"] = state_split[0]
    if all_data is None:
        all_data = df_t13
    else:
        all_data = pd.concat([all_data, df_t13])
    return all_data

def _establish_last_name(name, father_name):
    if name is np.nan:
        name = "FNU"
    if father_name is np.nan:
        father_name = "FNU"
    if len(name.split()) > 1 :
        last_name = name.split()[-1]
    else:
        if len(father_name.split()) > 1:
            last_name = father_name.split()[-1]
        else:
            last_name = "LNU"
    return last_name

def establish_last_name(df):
    print("Cleaning Data, processing last names")
    df["last_name"] = df.apply(lambda x: _establish_last_name(x["elector_name"], x["father_or_husband_name"]), axis=1)
    df = df.drop(["father_or_husband_name", "elector_name"],  axis=1)
    return df


def export_csv_gz(scraped_data, base_dir, chunk_num):
    scraped_data.to_csv(
        os.path.join(base_dir, f"instate_parsed_{chunk_num}.csv.gz"),
        header=False,
        index=False,
        compression="gzip",
    )
    chunk_num += 1
    scraped_data = None
    print(f"Writing chunk number : {chunk_num} in {base_dir}")
    return chunk_num, scraped_data


if __name__ == "__main__":
    scraped_data = None
    base_dir = "/data/in-rolls/parsed/"
    chunk_size = 1000000
    write_chunk = 10000000
    chunk_num = 0
    unsupported_states = ["himachal"]

    # # "*csv"
    df_csv = scrape_csv_folder(base_dir)
    df_csv = establish_last_name(df_csv)
    scraped_data = pd.concat([scraped_data, df_csv])

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
            df_7z = scrape_csv_folder(os.path.join(base_dir, f))
            df_7z = establish_last_name(df_7z)
            scraped_data = pd.concat([scraped_data, df_7z])
            if scraped_data.shape[0] > write_chunk:
                chunk_num, scraped_data = export_csv_gz(
                    scraped_data, base_dir, chunk_num
                )

    # *.gz.csv, pre-req: use scripts/concatenate.py to merge .partaa, .partab, etc files
    for base_path in sorted(glob(os.path.join(base_dir, "*.csv.gz"))):
        all_data = None
        state_split = re.split("[- _ : +]", os.path.basename(base_path))
        state = state_split[0]
        if state in unsupported_states:
            print(f"Skipping unsupported state: {state}")
        else:
            df_test = pd.read_csv(base_path, index_col=0, nrows=1)
            columns = df_test.columns.tolist()
            for df in pd.read_csv(base_path, chunksize=chunk_size):
                print(f"Processing, gz folder,  state: {state}, chunk {chunk_size}")
                df_gz = scrape_gz_chunk(df, state_split)
                df_gz = establish_last_name(df_gz)
                scraped_data = pd.concat([scraped_data, df_gz])
                if scraped_data.shape[0] > write_chunk:
                    chunk_num, scraped_data = export_csv_gz(
                        scraped_data, base_dir, chunk_num
                    )
