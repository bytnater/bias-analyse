'''
This file contains the class for the metrics 'Predictive parity' and 'Conditional Use Accuracy Equality'

Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism
prediction instruments. Big data, 5 (2), 153–163.

Verma, S., & Rubin, J. (2018). Fairness definitions explained. Proceedings of the International
Workshop on Software Fairness,
url: https://doi.org/10.1145/3194770.3194776

author: Casper K.
date: Jun 2025
'''

import torch
import plotly.graph_objects as go

class Predictive_value_metrics:
    def __init__(self, dataset, params):
        """
        Parameters:
            - dataset: tensor, the dataset of features
            - ground_truth_column: str, name of the ground truth column
            - prediction_column: str, name of the predicction column
            - protected_values: list, list of bools where protected values are true
        """

        self.dataset = dataset

        # Extract prediction column and binarize.
        prediction = self._get_prediction_column(params)
        prediction = (prediction > 0.7).float().squeeze()
        
        # Extract ground truth labels
        # Currently 70% certainty of model indicates a 1 prediction but can be changed
        ground_truth = self._get_ground_truth_column(params)
        
        # Get list of protected attributes from i2c and the provided protection mask
        protected_values = params.get('protected_values', torch.zeros(len(dataset.i2c), dtype=bool))
        self.protected_attributes = [
            name for name, is_protected in zip(dataset.i2c, protected_values)
            if is_protected
        ]
        
        # Tolerance threshold for acceptable fairness deviation (not directly used here)
        self.threshold = 0.1 #Accepted difference of fairness

        # Compute PPV and NPV per value of each protected attribute
        self.results = {}
        for attr in self.protected_attributes:
            col_idx = self.dataset.c2i[attr]
            protected_col = self.dataset.data[:, col_idx]
            unique_values = torch.unique(protected_col)

            PPV_dict = {}  # Positive Predictive Value per group
            NPV_dict = {}  # Negative Predictive Value per group

            for val in unique_values:
                # Filter data to rows matching this protected group value
                mask = protected_col == val
                preds = prediction[mask]
                truth = ground_truth[mask]
                if len(preds) > 0:
                    # Compute confusion matrix components
                    TP = ((preds == 1) & (truth == 1)).sum()
                    TN = ((preds == 0) & (truth == 0)).sum()
                    FP = ((preds == 1) & (truth == 0)).sum()
                    FN = ((preds == 0) & (truth == 1)).sum()

                    # Predictive Parity and Conditional Use Accuracy
                    ppv = TP / (TP + FP)
                    npv = TN / (TN + FN)
                else:
                    ppv = float('nan')
                    npv = float('nan')
                
                # Store per group value (cast to int for clean x-axis labels)    
                PPV_dict[int(val)] = ppv
                NPV_dict[int(val)] = npv
            
            # Save per-attribute result
            self.results[attr] = {
                "ppv": PPV_dict,
                "npv": NPV_dict
            }

    def _get_column(self, feature):
        # Extract a single column by index
        return self.dataset.data[:,feature].squeeze(-1)

    def _get_ground_truth_column(self, params):
        # Extract ground truth column name from parameters
        ground_truth_column = params.get('ground_truth_column', '')
        assert ground_truth_column != '', 'Predictive parity metrics needs a ground truth'
        return self.dataset.data[:,self.dataset.c2i[ground_truth_column]]

    def _get_prediction_column(self, params):
        # Extract prediction column name from parameters
        prediction_column = params.get('prediction_column', '')
        assert prediction_column != '', 'Predictive parity metrics needs a prediction'
        return self.dataset.data[:,self.dataset.c2i[prediction_column]]
    
    def show(self, raw_results=False) -> go.Figure:
        """
        Return either raw results or bar charts.
        """
        if raw_results:
            return self.results
        fig_list = []
        for attr_name in self.results:
            attr_data = self.results[attr_name]
            ppv = attr_data['ppv']
            npv = attr_data['npv']
            
            fig = go.Figure()
            
            # Add PPV bars
            fig.add_trace(go.Bar(
                name='PPV',
                x=[str(k) for k in ppv.keys()],
                y=list(ppv.values())
            ))

            # Add NPV bars
            fig.add_trace(go.Bar(
                name='NPV',
                x=[str(k) for k in npv.keys()],
                y=list(npv.values())
            ))

            # Layout config
            fig.update_layout(
                title=f'PPVs and NPVs by Group for "{attr_name}"',
                xaxis_title='Group',
                yaxis_title='PPVs and NPVs',
                yaxis=dict(range=[0, 1]),
                bargap=0.2
            )

            fig_list.append(fig)
        return fig_list
    
print('loaded predictive parity class')
