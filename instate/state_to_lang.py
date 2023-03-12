import pandas as pd

def state_to_lang(df, statecolname):
    state_lang = pd.read_csv("data/state_to_languages.csv")
    res = df.merge(state_lang, left_on=statecolname, right_on='state', how='left')
    return(res)
