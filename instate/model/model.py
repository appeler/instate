import pandas as pd
from glob import glob
import os

MIN_OCCURENCE = 3


def model_v0(df):
    return df[["last_name", "state"]].value_counts().reset_index(name="count")


def predict_v0(df, name):
    selected_name = df[df["last_name"] == name]
    selected_name["percent"] = (
        selected_name["count"] / selected_name["count"].sum()
    ) * 100
    return selected_name[["state", "percent"]].reset_index(drop=True)


def process_data(base_path):
    df = pd.read_csv(base_path)
    df = df[df.last_name != "LNU"]  # Remove last name unknows
    df = df[
        df.groupby("last_name")["last_name"].transform("count").ge(MIN_OCCURENCE)
    ]  # Remove all last names that occur less than 3 times
    return df


if __name__ == "__main__":
    base_dir = "/path/to/instate_data"
    fid = "instate_selected.csv.gz"
    # Test Prediction
    name = "DATTA"
    df = process_data(os.path.join(base_dir, fid))
    model = model_v0(df)
    print(predict_v0(model, name))
