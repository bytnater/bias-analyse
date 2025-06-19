import torch
import itertools

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
        protected_attributes = preset.get('protected_values', torch.zeros(len(self.dataset.i2c), dtype=bool)).nonzero().view(-1)  ## gains a list of indices, where the index encodes the name
        
        predictions = self.dataset.data[:,self.dataset.c2i[prediction_column]]  ## extract column with predictions from datset
        self.predictions = (predictions > 0.8).float().squeeze()
        #Currently 80% certainty of model indicates a 1 prediction but can be changed
        
        self.protected_attributes = [dataset.i2c[protected_attributes[i]] for i in range(len(protected_attributes))]  ## changes them to a list with names

        self.threshold = 0.1 #Accepted difference for statistical parity

    def check_statistical_parity(self):
        """
        c2i: Collum index
        :)
        steen papier schaar, go
        steen
        """
        results = {}
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

            results[attr] = {
                "group_probs": probs,
                "pairwise_differences": pair_diffs,
                "fair": is_fair
            }
            
        return results
    
    def show(self):
        # TODO, a funtion to represent data, via a simple message of a graph
        pass
