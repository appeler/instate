#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd

from pkg_resources import resource_filename

from utils import column_exists, fixup_columns, get_app_file_path, download_file


IN_ROLLS_DATA = {
    "v1": "https://dataverse.harvard.edu/api/v1/access/datafile/6979052",
}


class InRollsLnData:
    __df = None
    __model = None
    __year = None
    __tk = None
    __dataset = None

    @staticmethod
    def load_instate_data(dataset):
        data_fn = "instate_unique_ln_state_prop_{0:s}.csv.gz".format(dataset)
        data_path = get_app_file_path("instate", data_fn)
        if not os.path.exists(data_path):
            print("Downloading instate data from the server ({0!s})...".format(data_fn))
            if not download_file(IN_ROLLS_DATA[dataset], data_path):
                print("ERROR: Cannot download instate data file")
                return None
        else:
            print("Using cached instate data from local ({0!s})...".format(data_path))
        return data_path

    @classmethod
    def pred_last_state(cls, input):
        """
        Predict gender based on name
        Args:
            input (list of str): list of last name
        Returns:
            DataFrame: Pandas DataFrame with predictions
        """
        # load model
        if cls.__model is None:
            model_fn = resource_filename(__name__, "model")
            cls.__model = tf.keras.models.load_model(f"{model_fn}/instate_rmse")
        # create tokenizer
        if cls.__tk is None:
            cls.__tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            char_dict = {}
            for i, char in enumerate(alphabet):
                char_dict[char] = i + 1
            # Use char_dict to replace the tk.word_index
            cls.__tk.word_index = char_dict.copy()
            # Add 'UNK' to the vocabulary
            cls.__tk.word_index[cls.__tk.oov_token] = max(char_dict.values()) + 1

        input = [i.lower() for i in input]
        sequences = cls.__tk.texts_to_sequences(input)
        tokens = pad_sequences(sequences, maxlen=24, padding="post")

        results = cls.__model.predict(tokens)
        
        return pd.DataFrame(data={"name": input, "pred_state": results})

    @classmethod
    def last_state(cls, df, namecol, dataset="v1"):
        """Appends additional columns from state ratio data to the input DataFrame
        based on the last name.

        Removes the extra space. Checks if the name is the Indian electoral rolls data.
        If it is, outputs data from that row.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            namecol (str or int): Column's name or location of the name in
                DataFrame.
            state (str): The state name of Indian electoral rolls data to be used.
                (default is None for all states)

        Returns:
            DataFrame: Pandas DataFrame with 31 additional columns

        """

        if namecol not in df.columns:
            print("No column `{0!s}` in the DataFrame".format(namecol))
            return df

        df["__last_name"] = df[namecol].str.strip()
        df["__last_name"] = df["__last_name"].str.lower()

        if cls.__df is None or cls.__dataset != dataset:
            cls.__dataset = dataset
            data_path = InRollsLnData.load_instate_data(dataset)
            adf = pd.read_csv(data_path)
            cls.__df = adf
            cls.__df = cls.__df[["last_name"] + IN_ROLLS_COLS]
            cls.__df.rename(columns={"last_name": "__last_name"}, inplace=True)
        rdf = pd.merge(df, cls.__df, how="left", on="__last_name")

        return rdf

    @classmethod
    def state_to_lang(cls, df, statecolname):
        state_lang = pd.read_csv("data/state_to_languages.csv")
        res = df.merge(state_lang, left_on=statecolname, right_on='state', how='left')
        return(res)

    @staticmethod
    def list_states(dataset="v1"):
        data_path = InRollsLnData.load_instate_data(dataset)
        adf = pd.read_csv(data_path, usecols=["state"])
        return adf.state.unique()

last_state = InRollsLnData.last_state
pred_last_state = InRollsLnData.pred_last_state
state_to_lang = InRollsLnData.state_to_lang

