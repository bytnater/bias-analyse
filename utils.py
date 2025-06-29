import pandas as pd
import torch
from io import BytesIO

DATA_PATH_SYNTH = 'data/synth_data.csv'
SAVED_PRESET_PATH = 'presets/'
SAVED_DATASET_PATH = 'data/'
RESEVERD_PRESET = 'session_save.pt'


def load_file_to_torch(path=None, upload_widget=None):
    if upload_widget is not None:
        uploaded_file = upload_widget[0]
        content = uploaded_file.content
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(BytesIO(content))
        else:
            raise ValueError('File type not supported.')

    elif path is not None:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            raise ValueError('File type not supported.')
    else:
        raise ValueError("Either a file path or a file upload widget must be provided.")

    i2c = df.columns.tolist()
    c2i = {c: i for i, c in enumerate(i2c)}
    data_torch = torch.from_numpy(df.values)
    return data_torch, (i2c, c2i)

class Dataset():
    def __init__(self, PATH=None, upload_widget=None):
        data, (i2c, c2i) = load_file_to_torch(PATH, upload_widget)
        self.data = data
        self.i2c = i2c
        self.c2i = c2i

print('loaded utils')
