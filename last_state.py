import pandas as pd

def last_state(df, namecol):
	download_file()
	state_name = pd.read_csv("data/state_to_languages.csv")
    res = df.merge(state_name, left_on=namecol, right_on='state', how='left')
    return(res)
