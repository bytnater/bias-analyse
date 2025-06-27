'''
This file contains the class for the metrics '(Un)conditional Statistical Parity'

author: Casper K.
date: Jun 2025
'''

import torch
import plotly.graph_objects as go

class statistical_parity:
    def __init__(self, dataset, params):
        """
        Parameters:
            - dataset: tensor, the dataset of features
            - prediction_column: str, name of the predicction column
            - protected_values: list, list of bools where protected values are true
            - condition: dict, dictionary of attributes and specific value you want the individuals to have, leave empty for unconditional metric form
        """
        self.dataset = dataset


        prediction_column = params.get('prediction_column', '')
        assert prediction_column != '', 'This metric needs a prediction'
        self.condition = params.get('condition', dict())
        predictions = self.dataset.data[:,self.dataset.c2i[prediction_column]]  ## extract column with predictions from datset
        self.predictions = (predictions > 0.7).float().squeeze()
        #Currently 80% certainty of model indicates a 1 prediction but can be changed
        
        protected_values = params.get('protected_values', torch.zeros(len(dataset.i2c), dtype=bool))
        self.protected_attributes = [
            name for name, is_protected in zip(dataset.i2c, protected_values)
            if is_protected
        ]

        #calculations
        self.results = {}
        for attr in self.protected_attributes:
            col_idx = self.dataset.c2i[attr]
            protected_col = self.dataset.data[:, col_idx]

            condition_mask = torch.ones(len(self.dataset.data), dtype=bool)
            for cond_attr, cond_val in  self.condition.items():
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
                    proportion = preds.mean()
                else:
                    proportion = float('nan')
                probs[int(val)] = proportion

            self.results[attr] = {
                "group_probs": probs
            }

    def _get_column(self, feature):
        return self.dataset.data[:,feature].squeeze(-1)

    def _get_prediction_column(self, params):
        prediction_column = params.get('prediction_column', '')
        assert prediction_column != '', 'Statistical parity metrics needs a prediction'
        return self.dataset.data[:,self.dataset.c2i[self.prediction_column]]
    
    def show(self, raw_results=False) -> go.Figure:
        if raw_results:
            return self.results
        fig_list = []
        for attr_name in self.results:
            attr_data = self.results[attr_name]
            probs = attr_data['group_probs']
            
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[str(k) for k in probs.keys()],
                y=list(probs.values()),
            ))
            fig.update_layout(
                title=f'(Conditional) Positive Prediction Rate by Group for "{attr_name}"',
                xaxis_title='Group',
                yaxis_title='(Conditional) Positive Prediction Rate',
                yaxis=dict(range=[0, 1]),
                bargap=0.2
            )

            fig_list.append(fig)
        return fig_list
    
print('loaded Statistical parity class')
