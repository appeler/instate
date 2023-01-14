import pandas as pd
from glob import glob
import os
import re


def scrape_csv_folder(base_dir):
    all_data = None
    for fn in sorted(glob(os.path.join(base_dir, "*.csv"))):
        state = os.path.basename(fn).split(".")[0].split("-")[0]
        df = pd.read_csv(fn)
        if df.columns.isin(["name"]).any():
            df = pd.read_csv(fn, usecols=["name", "state"])
            df = df.rename(columns={"name": "elector_name"})
        else:
            df = pd.read_csv(fn, usecols=["elector_name", "state"])
        df["state"] = state
        if "guj" in state.split():
            print(f"Skipping non-supported language {state}")
        else:
            if all_data is None:
                all_data = df
            else:
                all_data = pd.concat([all_data, df])
            print(state, df.columns)
    return all_data


def scrape_gz_folder(base_path, chunk_size):
    all_data = None
    state_split = re.split("[- _ : +]", os.path.basename(base_path))
    state = state_split[0]
    for df in pd.read_csv(base_path, chunksize=chunk_size):
        if "clean" in state_split:
            df_t13 = df[["elector_name_t13n", "state"]]
            df_t13 = df_t13.rename(columns={"elector_name_t13n": "elector_name"})
        else:
            df_t13 = df[["elector_name", "state"]]
        df_t13["state"] = state
        if all_data is None:
            all_data = df_t13
        else:
            all_data = pd.concat([all_data, df_t13])
        print(state, df_t13.columns)
    return all_data


def export_csv_gz(scraped_data, base_dir, chunk_num):
    scraped_data.to_csv(
        os.path.join(base_dir, f"insate_parsed_{chunk_num}.csv.gz"),
        header=False,
        index=False,
        compression="gzip",
    )
    chunk_num += 1
    scraped_data = None
    print(f"Writing chunk number : {chunk_num}")
    return chunk_num, scraped_data


if __name__ == "__main__":
    scraped_data = None
    base_dir = "/data/in-rolls/parsed/"
    chunk_size = 1000000
    chunk_num = 0

    # "*csv"
    df_csv = scrape_csv_folder(base_dir)
    scraped_data = pd.concat([scraped_data, df_csv])

    # *.7z, pre-req: Extract .7z file to a folder manually using system unzipper
    for f in [
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]:
        df_7z = scrape_csv_folder(os.path.join(base_dir, f))
        scraped_data = pd.concat([scraped_data, df_7z])
        if scraped_data.shape[0] > chunk_size:
            chunk_num, scraped_data = export_csv_gz(scraped_data, base_dir, chunk_num)

    # *.gz.csv, pre-req: use scripts/concatenate.py to merge .partaa, .partab, etc files
    for base_path in sorted(glob(os.path.join(base_dir, "*.csv.gz"))):
        df_gz = scrape_gz_folder(base_path, chunk_size)
        scraped_data = pd.concat([scraped_data, df_gz])
        if scraped_data.shape[0] > chunk_size:
            chunk_num, scraped_data = export_csv_gz(scraped_data, base_dir, chunk_num)
