import pandas as pd
import torch

DATA_PATH_SYNTH = 'data/synth_data.csv'
SAVED_PRESET_PATH = 'presets/'
SAVED_DATASET_PATH = 'data/'
RESEVERD_PRESET = 'session_save.pt'

def load_csv_to_torch(path=DATA_PATH_SYNTH):
    df = pd.read_csv(path)
    i2c = df.columns.tolist()
    c2i = {c: i for i, c in enumerate(i2c)}
    data_torch = torch.from_numpy(df.values)
    return data_torch, (i2c, c2i)

class Dataset():
    def __init__(self, PATH):
        data, (i2c, c2i) = load_csv_to_torch(PATH)
        self.data = data
        self.i2c = i2c
        self.c2i = c2i