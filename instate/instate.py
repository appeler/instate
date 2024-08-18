#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tarfile
import json

import pandas as pd
import torch
from Levenshtein import distance

from typing import Union, List
from .models.model_lang import LanguagePredictor

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
    def load_data(file_name: str) -> Union[str, os.PathLike]:
        data_path = get_app_file_path("instate", file_name)
        if not os.path.exists(data_path):
            print(f"Downloading instate data from the server ({file_name})...")
            if not download_file(IN_ROLLS_DATA[file_name], data_path):
                print("ERROR: Cannot download instate data file")
                return None
        else:
            print(f"Using cached instate data from local ({data_path})...")
        return data_path

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
    def list_states(dataset: str = "v1") -> List[str]:
        data_path = InRollsLnData.load_instate_data(dataset)
        adf = pd.read_csv(data_path, usecols=["state"])
        return adf.state.unique()

    @staticmethod
    def lookup_lang(df: pd.DataFrame, lastnamecol: str) -> pd.DataFrame:
        if not column_exists(df, lastnamecol):
            return df
        data_file_name = "lastname_langs_india"
        data_path = get_app_file_path("instate", data_file_name)
        if not os.path.exists(data_path+".csv"):
            data_dir = os.path.dirname(__file__)
            gz_path = os.path.join(data_dir, 'data', f'{data_file_name}.csv.tar.gz')  
            print(f"Extracting {gz_path} to {data_path}")
            with tarfile.open(gz_path, "r:gz") as tar:
                tar.extract(f"{data_file_name}.csv", data_path)
        name_to_lang = pd.read_csv(f"{data_path}/{data_file_name}.csv")
        langs = name_to_lang.columns[1:]
        final = []
        for lastname in df[lastnamecol]:
            # use edit distance find top 3 nearest names
            distances = name_to_lang['last_name'].apply(lambda x: distance(lastname, x))
            nearest_lang = name_to_lang.loc[distances.nsmallest(3).index, langs].sum().idxmax()
            final.append(nearest_lang)
        
        # append final to df
        df['predicted_lang'] = final
        return df


    @staticmethod
    # do inference based on last_name
    def infer(lastname, char2idx, idx2lang, model, device):
        with torch.no_grad():
            last_name_indices = [char2idx[char] for char in lastname]
            last_name_tensor = torch.tensor(last_name_indices, dtype=torch.long).unsqueeze(0).to(device)
            lengths = torch.tensor([len(lastname)], dtype=torch.long)
            out1, out2, out3 = model(last_name_tensor, lengths)
            pred_first_lang = torch.argmax(out1, dim=1)
            pred_second_lang = torch.argmax(out2, dim=1)
            pred_third_lang = torch.argmax(out3, dim=1)
            # if second lang matches first, go to the next argmax
            if pred_second_lang == pred_first_lang:
                pred_second_lang = torch.topk(out2, k=2, dim=1)[1][0][1]
            if pred_third_lang == pred_first_lang or pred_third_lang == pred_second_lang:
                pred_third_lang = torch.topk(out3, k=3, dim=1)[1][0][1]
            return [idx2lang[pred_first_lang.item()], idx2lang[pred_second_lang.item()], idx2lang[pred_third_lang.item()]]


    @staticmethod
    def predict_lang(df: pd.DataFrame, lastnamecol: str) -> pd.DataFrame:
        if not column_exists(df, lastnamecol):
            return df
        
        data_dir = os.path.dirname(__file__)
        langs_file = os.path.join(data_dir, 'data', "langs.txt")
        with open(langs_file) as f:
            langs = f.read().splitlines()

        char2idx_file = os.path.join(data_dir, 'data', "char2idx.json")
        with open(char2idx_file) as f:
            char2idx = json.load(f)

        idx2char = {idx: char for char, idx in char2idx.items()}
        lang2idx = {lang: idx for idx, lang in enumerate(langs)}
        idx2lang = {idx: lang for lang, idx in lang2idx.items()}

        vocab_size = len(char2idx)
        embedding_dim = 50
        hidden_dim = 128  # Number of features in the hidden state of LSTM
        num_languages = len(langs)  

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LanguagePredictor(vocab_size, embedding_dim, hidden_dim, num_languages)
        model.to(device)

        model_file = os.path.join(data_dir, 'data', "state_lang_labels.pt")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_file))
        else:
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        model.eval()

        # for every last name, predict the language
        pred_langs = []
        for lastname in df[lastnamecol]:
            pred_langs.append(InRollsLnData.infer(lastname, char2idx, idx2lang, model, device))

        df['predicted_lang'] = pred_langs
        return df
        

last_state = InRollsLnData.last_state
pred_last_state = InRollsLnData.pred_last_state
state_to_lang = InRollsLnData.state_to_lang
list_states = InRollsLnData.list_states
lookup_lang = InRollsLnData.lookup_lang
predict_lang = InRollsLnData.predict_lang

