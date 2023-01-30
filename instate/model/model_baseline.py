import os

import pandas as pd
from instate.model.model_dnn import (
    process_data,
    prepare_test_data,
    split_tr_te,
)
import random
import difflib

# Order + States correct
# States correct
# top-label in the set of top-3


def predict_baseline(X_new_tr, y_new_tr, X_te, y_te, n_samples=1000):
    top_3 = []
    X_te_split = random.sample(list(X_te), n_samples)
    for i, _name in enumerate(X_te_split):
        print(f"Processing: {i}/{n_samples} sample")
        y_pred = difflib.get_close_matches(_name, X_new_tr, n=1)[0]
        top_3_pred = y_new_tr[y_pred][:3]
        top_1_gt = y_te[_name][0]
        if top_1_gt in top_3_pred:
            top_3.append(1)
        else:
            top_3.append(0)

    return sum(top_3) / len(top_3) * 100


if __name__ == "__main__":
    base_dir = "/Users/dhingratul/Documents/instate_data"
    fid = "instate_processed.csv.gz"
    # Data
    print("Processing data")
    df = process_data(os.path.join(base_dir, fid))
    training_data, test_data = split_tr_te(df)
    X_tr, y_tr = prepare_test_data(training_data)
    X_te, y_te = prepare_test_data(test_data)
    # # Eval
    top_3 = []
    for _x in X_te:
        y_gt = y_te[_x][0]
        y_pred = predict_baseline(_x, y_tr)
        if y_gt in y_pred:
            top_3.append(1)
        else:
            top_3.append(0)
