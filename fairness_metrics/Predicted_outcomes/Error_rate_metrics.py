'''
This file contains the class for the metrics 'Equalised Odds' and 'Equal Opportunity'

Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Ad-
vances in neural information processing systems, 29.

Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2017). Algorithmic decision making and the cost of fairness. 
Proceedings of the 23rd acm sigkdd international conference on knowledge discovery and data mining, 797–806.

Verma, S., & Rubin, J. (2018). Fairness definitions explained. Proceedings of the International
Workshop on Software Fairness,
url: https://doi.org/10.1145/3194770.3194776

author: Casper K.
date: Jun 2025
'''

import torch
import plotly.graph_objects as go

class Error_rate_metrics:
    def __init__(self, dataset, params):
        """
        Parameters:
            - dataset: tensor, the dataset of features
            - ground_truth_column: str, name of the ground truth column
            - prediction_column: str, name of the prediction column
            - protected_values: list, list of bools where protected values are True
        """

        self.dataset = dataset

        # Get and prediction values
        prediction = self._get_prediction_column(params)
        prediction = (prediction > 0.7).float().squeeze()
        
        # Get ground truth labels
        ground_truth = self._get_ground_truth_column(params)
        
        ## -- Currently 70% certainty of model indicates a 1 prediction but can be changed -- 
        
        protected_values = params.get('protected_values', torch.zeros(len(dataset.i2c), dtype=bool))
        self.protected_attributes = [
            name for name, is_protected in zip(dataset.i2c, protected_values)
            if is_protected
        ]

        # Tolerance threshold for fairness deviatio 
        self.threshold = 0.1 #Accepted difference of fairness

        # Calculate FPR and FNR for each protected group
        self.results = {}
        for attr in self.protected_attributes:
            col_idx = self.dataset.c2i[attr]
            protected_col = self.dataset.data[:, col_idx]
            unique_values = torch.unique(protected_col)

            fpr_dict = {}
            fnr_dict = {}

            for val in unique_values:
                # Select rows that belong to the current group
                mask = protected_col == val
                preds = prediction[mask]
                truth = ground_truth[mask]
                if len(preds) > 0:
                    # Confusion matrix components
                    TP = ((preds == 1) & (truth == 1)).sum()
                    TN = ((preds == 0) & (truth == 0)).sum()
                    FP = ((preds == 1) & (truth == 0)).sum()
                    FN = ((preds == 0) & (truth == 1)).sum()

                    # Compute rates
                    fpr = FP / (FP + TN)
                    fnr = FN / (FN + TP)
                else:
                    fpr = float('nan')
                    fnr = float('nan')
                
                # Store results per group value
                fpr_dict[int(val)] = fpr
                fnr_dict[int(val)] = fnr

            # Store FPR and FNR for this protected attribute
            self.results[attr] = {
                "fpr": fpr_dict,
                "fnr": fnr_dict
            }

    def _get_column(self, feature):
        # To extract a feature column from the dataset
        return self.dataset.data[:,feature].squeeze(-1)

    def _get_ground_truth_column(self, params):
        # Retrieve ground truth column from dataset
        ground_truth_column = params.get('ground_truth_column', '')
        assert ground_truth_column != '', 'Error-based metrics needs a ground truth'
        return self.dataset.data[:,self.dataset.c2i[ground_truth_column]]

    def _get_prediction_column(self, params):
        # Retrieve prediction column from dataset
        prediction_column = params.get('prediction_column', '')
        assert prediction_column != '', 'Error-based metrics needs a prediction'
        return self.dataset.data[:,self.dataset.c2i[prediction_column]]
    
    def show(self, raw_results=False) -> go.Figure:
        """
        Returns visual bar chart(s) of FPR and FNR for each protected attribute.
        If raw_results=True, return data instead.
        """
        if raw_results:
            return self.results
        fig_list = []
        for attr_name in self.results:
            attr_data = self.results[attr_name]
            fpr = attr_data['fpr']
            fnr = attr_data['fnr']
            
            fig = go.Figure()
            
            # Add bars for FPR
            fig.add_trace(go.Bar(
                name='FPR',
                x=[str(k) for k in fpr.keys()],
                y=list(fpr.values())
            ))

            # Add bars for FNR
            fig.add_trace(go.Bar(
                name='FNR',
                x=[str(k) for k in fnr.keys()],
                y=list(fnr.values())
            ))

            # Layout styling
            fig.update_layout(
                title=f'FPRs and FNRs by Group for "{attr_name}"',
                xaxis_title='Group',
                yaxis_title='FPRs and FNRs',
                yaxis=dict(range=[0, 1]),
                bargap=0.2
            )

            fig_list.append(fig)

        return fig_list
    
print('loaded error-based class')