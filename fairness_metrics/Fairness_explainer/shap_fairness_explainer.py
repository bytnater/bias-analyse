import pandas as pd
import torch
import shap
import numpy as np
from xgboost import XGBRegressor

class shap_fairness_explainer():
    """
    Wraps SHAP explainability for fairness auditing.

    Parameters:
        dataset: Dataset object with .data, .i2c, .c2i
        parameters: dict with:
            - "model": trained model (e.g. XGBRegressor)
            - "features_model": list of feature names used in the model
    """
    def __init__(self, dataset, parameters):
        self.dataset = dataset
        self.model = parameters["model"]
        self.feature_names = parameters["features_model"]

        # Map feature names to column indices
        self.feature_indices = [dataset.c2i[f] for f in self.feature_names]

        # Extract feature matrix as numpy array for SHAP
        self.X = dataset.data[:, self.feature_indices].numpy()

        # Create SHAP explainer and compute SHAP values
        self.explainer = shap.Explainer(self.model, self.X)
        self.shap_values = self.explainer(self.X)

    def feature_importance(self, max_display=10):
        """
        Show violin plot of SHAP values (feature importance per instance).
        """
        shap.plots.violin(self.shap_values, max_display=max_display)


# ---------- EXAMPLE USAGE ---------- #


PATH = "data/synth_data_preds.csv"

def load_csv_to_torch(path=PATH):
    """
    Load CSV into a PyTorch tensor and create column index mappings.
    
    Returns:
        data_torch: Tensor of data values
        i2c: List of column names
        c2i: Dictionary mapping column name to index
    """
    df = pd.read_csv(path)
    i2c = df.columns.tolist()
    c2i = {c: i for i, c in enumerate(i2c)}
    data_torch = torch.from_numpy(df.values)
    return data_torch, (i2c, c2i)

class Dataset():
    """
    Simple dataset wrapper to hold data and metadata.
    """
    def __init__(self, PATH):
        data, (i2c, c2i) = load_csv_to_torch(PATH)
        self.data = data          # PyTorch tensor with all values
        self.i2c = i2c            # List: index to column name
        self.c2i = c2i            # Dict: column name to index


# Load dataset from CSV
data = Dataset(PATH)

# Define which features are used in the model
features = [
    "competentie_omgaan_met_verandering_en_aanpassen", 
    "competentie_onderzoeken",
    "adres_aantal_woonadres_handmatig"
]

# Prepare feature and label matrices for training
X_indices = [data.c2i[f] for f in features]           # Feature column indices
y_index = data.c2i["predictions"]                     # Target column index

X = data.data[:, X_indices].numpy()
y = data.data[:, y_index].numpy()

# Train a regression model (XGBoost)
model = XGBRegressor()
model.fit(X, y)

# Wrap the model and data into the SHAP fairness explainer
param = {
    "model": model,
    "features_model": features
}
explainer = shap_fairness_explainer(data, param)

# Visualize feature influence via SHAP violin plot
explainer.feature_importance()
