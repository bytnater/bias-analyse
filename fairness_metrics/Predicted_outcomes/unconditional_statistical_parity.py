import torch
import itertools

class statistical_parity:
    def __init__(self, data, predictions, protected_attributes):
        """
        data: dataset Tensor
        predictions: Tensor of model predictions
        protected_attributes: List of protected attribute names
        """
        self.data = data
        self.predictions = (predictions > 0.8).float().squeeze()
        #Currently 80% certainty of model indicates a 1 prediction but can be changed
        self.protected_attributes = protected_attributes
        self.threshold = 0.1 #Accepted difference for statistical parity

    def check_statistical_parity(self, c2i):
        results = {}
        for attr in self.protected_attributes:
            col_idx = c2i[attr]
            protected_col = self.data[:, col_idx]

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

            results[attr] = {
                "group_probs": probs,
                "pairwise_differences": pair_diffs,
                "fair": is_fair
            }
            
        return results