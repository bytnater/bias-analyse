import pandas as pd
import torch

DATA_PATH_SYNTH = 'data/synth_data.csv'

def load_csv_to_torch(path=DATA_PATH_SYNTH):
    df = pd.read_csv(path)
    i2c = df.columns.tolist()
    c2i = {c: i for i, c in enumerate(i2c)}
    data_torch = torch.from_numpy(df.values)
    return data_torch, (i2c, c2i)


