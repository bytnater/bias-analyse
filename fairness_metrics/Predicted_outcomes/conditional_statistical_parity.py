import torch
import itertools
import plotly.graph_objects as go

class conditional_statistical_parity:
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
        condition = preset.get('condition', dict())
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

            condition_mask = torch.ones(len(self.dataset.data), dtype=bool)
            for cond_attr, cond_val in condition.items():
                cond_idx = self.dataset.c2i[cond_attr]
                condition_mask &= (self.dataset.data[:, cond_idx] == cond_val)

            filtered_protected_col = protected_col[condition_mask]
            filtered_predictions = self.predictions[condition_mask]

            unique_values = torch.unique(filtered_protected_col)

            probs = {}
            for val in unique_values:
                mask = filtered_protected_col == val
                preds = filtered_predictions[mask]
                if len(preds) > 0:
                    proportion = preds.mean().item()
                else:
                    proportion = float('nan')  # Handle empty groups
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
            
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[str(k) for k in probs.keys()],
                y=list(probs.values()),
            ))
            fig.update_layout(
                title=f'Conditional Positive Prediction Rate by Group for "{attr_name}"',
                xaxis_title='Group',
                yaxis_title='Positive Prediction Rate',
                yaxis=dict(range=[0, 1]),
                bargap=0.2
            )

            fig.show()