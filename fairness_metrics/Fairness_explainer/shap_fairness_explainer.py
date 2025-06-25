import pandas as pd
import torch
import shap
import numpy as np
from xgboost import XGBRegressor


PATH = "data/synth_data_preds.csv"
def load_csv_to_torch(path=PATH):
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


class shap_fairness_explainer():
    def __init__(self, dataset, parameters):
        self.dataset = dataset
        self.model = parameters["model"]
        self.feature_names = parameters["features"]

        self.feature_indices = [dataset.c2i[f] for f in self.feature_names]

        self.X = dataset.data[:, self.feature_indices].numpy()

        self.explainer = shap.Explainer(self.model, self.X)
        self.shap_values = self.explainer(self.X)

    def feature_importance(self, max_display=10):
        shap.plots.bar(self.shap_values, max_display=max_display)




# ------------ EXAMPLE USAGE ------------ #

PATH = "data/synth_data_preds.csv"
data = Dataset(PATH)

features = [
    "competentie_omgaan_met_verandering_en_aanpassen", 
    "competentie_onderzoeken",
    "adres_aantal_woonadres_handmatig"
]

X_indices = [data.c2i[f] for f in features]
y_index = data.c2i["predictions"]

X = data.data[:, X_indices].numpy()
y = data.data[:, y_index].numpy()

model = XGBRegressor()
model.fit(X, y)

param = {
    "model": model,
    "features": features
}

explainer = shap_fairness_explainer(data, param)
explainer.feature_importance()
