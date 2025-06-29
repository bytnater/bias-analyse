'''
This file includes a class for the fairness metrics 'test-fairness' and 'well-calibration'

reference:
Chouldechova, A. (2016). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments.
url: https://arxiv.org/abs/1610.07524

Verma, S., & Rubin, J. (2018). Fairness definitions explained. Proceedings of the International
Workshop on Software Fairness,
url: https://doi.org/10.1145/3194770.3194776


author: Mick
date: Jun 2025
'''

import torch
import numpy as np
import plotly.graph_objects as go

class well_calibration:
    def __init__(self, dataset, params):
        '''
        Parameters:
            - dataset: tensor, the dataset of features
            - ground_truth_column: str, name of the ground truth column
            - prediction_column: str, name of the predicction column
            - protected_values: list, list of bools where protected values are true
        '''
        self.dataset = dataset
        self.params = params

        # Determine which features are protected
        check_features = params.get('protected_values', torch.zeros(len(self.dataset.i2c), dtype=bool))
        indices = check_features.nonzero()


        # calculation
        outcome_per_feature = []
        for feature in indices:
            # Extract the column for the protected attribute
            feature_column = self._get_column(feature)
            ground_truth = self._get_ground_truth_column(params)
            prediction = self._get_prediction_column(params)

            # Discretize prediction scores into bins 0-10
            prediction = torch.round(prediction / 0.1)  ## bins the predictions into bins from 0 to 10
            
            # Discretize prediction scores into bins 0-10
            unique_feature_values = feature_column.unique()
            calibration_data = []

            for feature_value in unique_feature_values:

                # Select all samples belonging to this group
                indices_feature = (feature_column == feature_value).nonzero().squeeze(-1)
                
                # Create bins 0–10 to track calibration
                bins = np.arange(0,10.5,1)
                data = {key: [0, 0] for key in bins}

                for index in indices_feature:
                    key = int(prediction[index])
                    data[key][1] += 1
                    if ground_truth[index] == 1:
                        data[key][0] += 1

                # Convert raw counts into calibrated fractions (i.e., empirical accuracy per bin)
                refit_data = [feature_value.item()]
                for data_key, data_item in data.items():
                    fraction = data_item[0]/data_item[1] if data_item[1] else 0
                    refit_data.append(fraction)

                calibration_data.append(refit_data)

            # Convert group data to tensor format for storage and visualization
            calibration_data = torch.tensor(calibration_data)
            outcome_per_feature.append((self.dataset.i2c[feature], calibration_data))
        
        # Store final result per protected feature
        self.results = outcome_per_feature

    def _get_column(self, feature):
        # Extract the data column for a given feature index
        return self.dataset.data[:,feature].squeeze(-1)

    def _get_ground_truth_column(self, params):
        # Load ground truth column based on parameter key
        ground_truth_column = params.get('ground_truth_column', '')
        assert ground_truth_column != '', 'Test-fairness metrics needs a ground truth'
        return self.dataset.data[:,self.dataset.c2i[ground_truth_column]]

    def _get_prediction_column(self, params):
        # Extract a single column by index
        prediction_column = params.get('prediction_column', '')
        assert prediction_column != '', 'Test-fairness metrics needs a prediction'
        return self.dataset.data[:,self.dataset.c2i[prediction_column]]

    def show(self, raw_results=False) -> go.Figure:
        """
        Return either raw results or bar charts.
        """
        if raw_results:
            return self.results
        bins = np.arange(0,1.05,.1)
        fig_list = []
        for feature, data in self.results:
            fig = go.Figure()

            if len(data) > 5:
                  # If too many groups, aggregate groups into 3-4 segments for readability
                bin_ranges = (np.linspace(0,len(data)-1,4)+.5).astype(int)
                for i, ni in zip(bin_ranges[:-1], bin_ranges[1:]):
                    # Average the curves for groups in this segment
                    new_data = sum(data[i:ni,1:])/len(data[i:ni,1:])
                    fig.add_trace(go.Line(
                            name=f'from {data[i,0]} to {data[ni,0]}',
                            x=bins,
                            y=new_data
                        ))
            else:
                # Plot each group separately
                for group_data in data:
                    fig.add_trace(go.Line(
                        name=f'{group_data[0]}',
                        x=bins,
                        y=group_data[1:]
                    ))

            fig.update_layout(
                title=f'Test-fairness for "{feature}"',
                xaxis_title='Predicted probability',
                yaxis_title='probability',
                yaxis=dict(range=[0, 1]),
                bargap=0.2
            )

            fig_list.append(fig)
        return fig_list

print('loaded test-fairness class')
