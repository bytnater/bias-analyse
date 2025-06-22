import torch
import itertools
import matplotlib.pyplot as plt

class statistical_parity:
    def __init__(self, dataset, preset):
        """
        data: dataset Tensor
        predictions: Tensor of model predictions
        protected_attributes: List of protected attribute names

        dataset: contains data, i2c, c2i
        preset: contains all important data in a dict, note preset is a bad name for it but it is already all over the code
        """
        self.dataset = dataset


        prediction_column = preset.get('prediction_column', '')
        assert prediction_column != '', 'This metric needs a prediction'
        predictions = self.dataset.data[:,self.dataset.c2i[prediction_column]]  ## extract column with predictions from datset
        self.predictions = (predictions > 0.8).float().squeeze()
        #Currently 80% certainty of model indicates a 1 prediction but can be changed
        
        protected_values = preset.get('protected_values', torch.zeros(len(dataset.i2c), dtype=bool))
        self.protected_attributes = [
            name for name, is_protected in zip(dataset.i2c, protected_values)
            if is_protected
        ]

        self.threshold = 0.1 #Accepted difference of fairness

        #calculations
        self.results = {}
        for attr in self.protected_attributes:
            col_idx = self.dataset.c2i[attr]
            protected_col = self.dataset.data[:, col_idx]

            unique_values = torch.unique(protected_col)

            probs = {}
            for val in unique_values:
                mask = protected_col == val
                preds = self.predictions[mask]
                proportion = preds.mean().item()
                probs[int(val.item())] = proportion

            pair_diffs = {}
            for g1, g2 in itertools.combinations(probs.keys(), 2):
                diff = abs(probs[g1] - probs[g2])
                pair_diffs[(g1, g2)] = diff

            is_fair = all(pairwise_diff <= self.threshold for pairwise_diff in pair_diffs.values())

            self.results[attr] = {
                "group_probs": probs,
                "pairwise_differences": pair_diffs,
                "fair": is_fair
            }
    
    def show(self, raw_results=False):
        if raw_results:
            return self.results
        for attr_name in self.results:
            attr_data = self.results[attr_name]
            probs = attr_data['group_probs']
            plt.title(f'Positive Prediction Rate for "{attr_name}"')
            plt.bar(probs.keys(), probs.values())
            plt.show()