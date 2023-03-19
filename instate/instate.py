#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import pandas as pd
import torch
import torch.nn as nn

from typing import Union
from pkg_resources import resource_filename

from .utils import column_exists, get_app_file_path, download_file, _load_model, _pred_last_state
from .nnets import infer, GRU_net, GT_KEYS, n_letters, n_hidden

IN_ROLLS_DATA = {
    "v1": "https://github.com/appeler/instate/raw/main/data/instate_unique_ln_state_prop_v1.csv.gz",
}


IN_ROLLS_MODELS = {
    "gru": "https://dataverse.harvard.edu/api/v1/access/datafile/6981460",
}

class InRollsLnData:
    __df = None
    __model = None
    __year = None
    __dataset = None

    @staticmethod
    def load_instate_data(dataset: str) -> Union[str, os.PathLike]:
        data_fn = f"instate_unique_ln_state_prop_{dataset}.csv.gz"
        data_path = get_app_file_path("instate", data_fn)
        if not os.path.exists(data_path):
            print(f"Downloading instate data from the server ({data_fn})...")
            if not download_file(IN_ROLLS_DATA[dataset], data_path):
                print("ERROR: Cannot download instate data file")
                return None
        else:
            print(f"Using cached instate data from local ({data_path})...")
        return data_path

    @staticmethod
    def load_instate_model(model: str = "gru") -> Union[str, os.PathLike]:
        model_fn = "instate_gru.pth"
        model_path = get_app_file_path("instate", model_fn)
        if not os.path.exists(model_path):
            print(f"Downloading instate model from the server ({model_fn})...")
            if not download_file(IN_ROLLS_MODELS[model], model_path):
                print("ERROR: Cannot download instate model file")
                return None
        else:
            print(f"Using cached instate model from local ({model_path})...")
        return model_path

    @classmethod
    def pred_last_state(cls, df: pd.DataFrame, lastnamecol: str, k: int =3) -> pd.DataFrame:
        """
        Predict state based on name. Filters the dataframe to lastnames more than 2 chars, with 
        only English alphabets, strips extra spaces, and converts last names to lowercase. 
        Also drops duplicates.
        Args:
            df: pandas dataframe with the last name column
            lastnamecol: column name with the last name
            k: the number of states that should be returned (in order). default is 3.
        Returns:
            DataFrame: Pandas DataFrame with appended predictions
        """
        model_fn = "instate_gru.pth"
        model_path = get_app_file_path("instate", model_fn)

        if not column_exists(df, lastnamecol):
            return df

        if cls.__model is None:
            model_path = InRollsLnData.load_instate_model("gru")
            cls.__model = _load_model(model_path)

        df = df[df[lastnamecol].str.isalpha()]
        df = df[df[lastnamecol].str.contains('[a-z]',  na=False, case = False)]
        df = df[df[lastnamecol].str.len() > 2]
        df[lastnamecol] = df[lastnamecol].str.strip().str.lower()
        df.drop_duplicates(subset=[lastnamecol], inplace = True)
    
        pred_arr = []
        name_list = df[lastnamecol].to_list()
        for _name in name_list:
            pred_arr.append(_pred_last_state(cls.__model, _name, k=k))

        df["pred_state"] = pred_arr

        return df        
        
    @classmethod
    def last_state(cls, df: pd.DataFrame, lastnamecol: str, dataset:str = "v1") -> pd.DataFrame:
        """Appends additional columns from state data to the input DataFrame
        based on the last name. 

        Removes the extra space. Checks if the name is the Indian electoral rolls data.
        If it is, outputs data from that row. Drops duplicated last names.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            lastnamecol (str or int): Column's name or location of the name in
                DataFrame.
            state (str): The state name of Indian electoral rolls data to be used.
                (default is None for all states)

        Returns:
            DataFrame: Pandas DataFrame with 31 additional columns

        """

        if not column_exists(df, lastnamecol):
            return df

        df["__last_name"] = df[lastnamecol].str.strip().str.lower()
        df.drop_duplicates(subset=[lastnamecol], inplace = True)

        if cls.__df is None or cls.__dataset != dataset:
            cls.__dataset = dataset
            data_path = InRollsLnData.load_instate_data(dataset)
            adf = pd.read_csv(data_path)
            cls.__df = adf
            cls.__df.rename(columns={"last_name": "__last_name"}, inplace = True)
        rdf = pd.merge(df, cls.__df, how = "left", on = "__last_name")

        return rdf

    @classmethod
    def state_to_lang(cls, df: pd.DataFrame, statecolname: str) -> pd.DataFrame:
        state_lang = pd.read_csv("data/state_to_languages.csv")
        res = df.merge(state_lang, left_on = statecolname, right_on = 'state', how = 'left')
        return(res)

    @staticmethod
    def list_states(dataset: str = "v1") -> list[str]:
        data_path = InRollsLnData.load_instate_data(dataset)
        adf = pd.read_csv(data_path, usecols=["state"])
        return adf.state.unique()

last_state = InRollsLnData.last_state
pred_last_state = InRollsLnData.pred_last_state
state_to_lang = InRollsLnData.state_to_lang
list_states = InRollsLnData.list_states

