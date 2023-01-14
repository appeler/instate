import pandas as pd
from glob import glob
import os


def scrape_csv_root(base_dir):
    all_data = None
    import pdb;pdb.set_trace()
    for fn in sorted(glob(base_dir + "*.csv")):
        import pdb;pdb.set_trace()
        state = os.path.basename(fn).split('.')[0].split('-')[0]
        df = pd.read_csv(fn)
        if df.columns.isin(['name']).any():
            df = pd.read_csv(fn, usecols=['name', 'state'])
            df = df.rename(columns={'name': 'elector_name'})
        else:
            df = pd.read_csv(fn, usecols=['elector_name', 'state'])
        df['state'] = state
        if all_data is None:
            all_data = df
        else:
            all_data = pd.concat([all_data, df])
        print(state, df.columns)
    import pdb;pdb.set_trace()
    return all_data

def scrape_csv_folder(base_dir):
    all_data = None
    import pdb;pdb.set_trace()
    for fn in sorted(glob(base_dir + "*.csv")):
        import pdb;pdb.set_trace()
        state = os.path.basename(fn).split('.')[0].split('-')[0]
        df = pd.read_csv(fn)
        if df.columns.isin(['name']).any():
            df = pd.read_csv(fn, usecols=['name', 'state'])
            df = df.rename(columns={'name': 'elector_name'})
        else:
            df = pd.read_csv(fn, usecols=['elector_name', 'state'])
        df['state'] = state
        if all_data is None:
            all_data = df
        else:
            all_data = pd.concat([all_data, df])
        print(state, df.columns)
    import pdb;pdb.set_trace()
    return all_data

if __name__ == "__main__":
    base_dir = "/data/in-rolls/parsed/"
    # #"*csv"
    # scrape_csv(base_dir)
    #*.7z
    # Extract .7z file to a folder manually
    for f in [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]:
        all_data = scrape_csv(os.path.join(base_dir, f))
