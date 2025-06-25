import torch
import itertools
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class LipschitzFairness:
    def __init__(self, dataset, parameters):
        """
            parameters:
                - sample_limit
                - prediction_column: str
                - feature_columns (for distance metric)
                - distance_metric: 
                    -- 'Manhattan Distance', 
                    -- 'Euclidean Distance', 
                    -- 'cosine'
                    -- callable
        """

        self.dataset = dataset
        self.prediction_column = parameters.get('prediction_column')
        # self.feature_columns = parameters.get('feature_columns')
        protected_values = parameters.get('protected_values', torch.zeros(len(dataset.i2c), dtype=bool))
        self.feature_columns = [
            name for name, is_protected in zip(dataset.i2c, protected_values)
            if is_protected
        ]

        
        assert self.prediction_column, "prediction_column missing"
        assert self.feature_columns, "feature_columns missing"

        self.distance_fn = self._get_distance_fn(parameters.get("distance_metric", "Euclidean Distance"))

        self.features = self._get_columns(self.feature_columns)
        self.features = self._normalize_features(self.features)
        self.predictions = self._get_column(self.prediction_column)

        # Sample limit
        self.sample_limit = parameters.get('sample_limit', 1000)
        self.indices = torch.randperm(len(self.predictions))[:self.sample_limit]
        self.total_pairs = len(self.indices) * (len(self.indices) - 1) // 2

        self.violations = []
        self._lipschitz_violations()

    def _get_column(self, col_name):
        return self.dataset.data[:, self.dataset.c2i[col_name]]

    def _get_columns(self, col_names):
        indices = [self.dataset.c2i[name] for name in col_names]
        return self.dataset.data[:, indices]
    
    def _normalize_features(self, X):
        return (X - X.mean(dim=0)) / X.std(dim=0)

    def _get_distance_fn(self, metric):
        
        if callable(metric):
            return metric
        elif metric == 'Manhattan Distance':
            return lambda x, y: torch.sum(torch.abs(x - y))
        elif metric == "Euclidean Distance":
            return lambda x, y: torch.norm(x - y, p=2)
        elif metric == "cosine":
            return lambda x, y: 1 - torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()
        else:
            raise ValueError(f"unsupported distance metric {metric}")
        
    def _lipschitz_violations(self):
        for i, j in itertools.combinations(self.indices.tolist(), 2):
            x_i, x_j = self.features[i], self.features[j]
            y_i, y_j = self.predictions[i].item(), self.predictions[j].item()

            d = self.distance_fn(x_i, x_j)
            prediction_diff = abs(y_i - y_j)

            if prediction_diff > d:
                self.violations.append({
                    'pair': (i, j),
                    'prediction_diff': prediction_diff,
                    'distance': d,
                    'amount': prediction_diff - d
                })

    def _violation_rate(self):
        return len(self.violations) / self.total_pairs
    
    def show(self, raw_results=False, bins=30):
        if raw_results:
            return self.violations
        
        if not self.violations:
                return "No violations to show."

        amounts = [v['amount'] for v in self.violations]

        # Compute histogram data manually
        hist_data = go.Histogram(
            x=amounts,
            nbinsx=bins,
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
        )

        layout = go.Layout(
            title="Distribution of Lipschitz Violation Amounts",
            xaxis=dict(title="Violation Amount"),
            yaxis=dict(title="Frequency"),
            bargap=0.05,
            template="simple_white"
        )

        fig = go.Figure(data=[hist_data], layout=layout)
        return [fig]

        # print("Violation count:", len(amounts))

        # plt.figure(figsize=(8, 4))
        # plt.hist(amounts, bins=bins, color='skyblue', edgecolor='black')
        # plt.title("Distribution of Lipschitz Violation Amounts")
        # plt.xlabel("Violation Amount)")
        # plt.ylabel("Frequency")
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.tight_layout()
        # plt.show()

# ---- Test ----

# PATH = "data/synth_data_preds.csv"
# def load_csv_to_torch(path=PATH):
#     df = pd.read_csv(path)
#     i2c = df.columns.tolist()
#     c2i = {c: i for i, c in enumerate(i2c)}
#     data_torch = torch.from_numpy(df.values)
#     return data_torch, (i2c, c2i)

# class Dataset():
#     def __init__(self, PATH):
#         data, (i2c, c2i) = load_csv_to_torch(PATH)
#         self.data = data
#         self.i2c = i2c  
#         self.c2i = c2i


# features = [
#         "competentie_omgaan_met_verandering_en_aanpassen", 
#         "competentie_onderzoeken",
#         "adres_aantal_woonadres_handmatig"
#         ]

# params = {
#     "prediction_column": "predictions",
#     "feature_columns": features,
#     "distance_metric": "cosine",
#     "sample_limit": 500
# }

# dataset = Dataset('data/synth_data_preds.csv')
# fairness = LipschitzFairness(dataset, parameters=params)

# fairness.show()

print('loaded similarity')