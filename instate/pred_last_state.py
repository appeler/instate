import pandas as pd
from instate.utils import download_file
from instate.models.nets import infer, GRU_net, GT_KEYS, n_letters, n_hidden, n_states
import torch
import torch.nn as nn


def _pred_last_state(model, name, k=3):
    # Expects name to be pre-processed, private function, not be used standalone
	output = infer(model, name)  # prediction
	_, indices = output.topk(k)  # get the top k predictions
	idx_list = indices.numpy().flatten().tolist()
	preds = [GT_KEYS[i] for i in idx_list]
	return preds

def _load_model(path):
	model = GRU_net(n_letters, n_hidden, n_states)
	model.load_state_dict(torch.load(path))
	return model

def pred_last_state(model_path, df, namecol, k=3):
	pred_arr = []
	model = _load_model(model_path)
	# download the model
	download_file()
	# preprocess the namecol
	df = df[df.namecol.str.isalpha()]
	df[namecol] = df[namecol].str.lower()
	df = df[df[namecol].str.contains('[a-z]',  na=False)]
	df = df[df.namecol.str.len() > 2]
	# run predict
	# TODO: Use .apply()
	name_list = df[namecol].to_list()
	for _name in name_list:
		pred_arr.append(_pred_last_state(model, _name, k=3))
	# append preds
	res = {'preds': pred_arr}
	df = df.append(pd.DataFrame(res))

	return df

if __name__ == "__main__":
    name = "Dutta"
    path = "path/to/instate_gru.pth"
    model = _load_model(path)
    print(_pred_last_state(model, name, k=3))


