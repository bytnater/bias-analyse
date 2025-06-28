import pandas as pd
import torch
from io import BytesIO

DATA_PATH_SYNTH = 'data/synth_data.csv'
SAVED_PRESET_PATH = 'presets/'
SAVED_DATASET_PATH = 'data/'
RESEVERD_PRESET = 'session_save.pt'


def load_csv_to_torch(path=None, upload_widget=None):
    if upload_widget is not None:
        uploaded_file = upload_widget[0]
        content = uploaded_file.content
        df = pd.read_csv(BytesIO(content))
    elif path is not None:
        df = pd.read_csv(path)
    else:
        raise ValueError("Either a file path or a file upload widget must be provided.")

    i2c = df.columns.tolist()
    c2i = {c: i for i, c in enumerate(i2c)}
    data_torch = torch.from_numpy(df.values)
    return data_torch, (i2c, c2i)

class Dataset():
    def __init__(self, PATH=None, upload_widget=None):
        data, (i2c, c2i) = load_csv_to_torch(PATH, upload_widget)
        self.data = data
        self.i2c = i2c
        self.c2i = c2i

print('loaded utils')
