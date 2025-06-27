'''
This file includes a class for the fairness metrics 'balance in positve/negative class'

author: Mick
date: Jun 2025
'''

import torch
import plotly.graph_objects as go

class balance_in_pos_neg:
    def __init__(self, dataset, params):
        '''
        Parameters:
            - dataset: tensor, the dataset of features
            - balance_pos: bool, a flag to show positive balance
            - balance_neg: bool, a flag to show negative balance
            - ground_truth_column: str, name of the ground truth column
            - prediction_column: str, name of the predicction column
            - protected_values: list, list of bools where protected values are true
        '''
        self.dataset = dataset
        self.calc_pos = params.get('balance_pos', True)
        self.calc_neg = params.get('balance_neg', True)
        assert self.calc_pos + self.calc_neg != 0, 'Select at least one metric'

        self.check_features = params.get('protected_values', torch.zeros(len(self.dataset.i2c), dtype=bool))
        indices = self.check_features.nonzero()
        # self.dataset.data[:,params['protected_values']]  ## filter all unimportent


        # calculation
        outcome_per_feature: list[tuple[str, list[int, float]]] = []
        for feature in indices:
            # print('feature i atm:',feature)
            feature_column = self._get_column(feature)
            ground_truth = self._get_ground_truth_column(params)
            prediction = self._get_prediction_column(params)
            unique_feature_values = feature_column.unique()
            # print('unique values', unique_feature_values)

            balance_data = []
            for feature_value in unique_feature_values:

                indices_feature = (feature_column == feature_value).nonzero().squeeze(-1)
                pos_total, neg_total = [0, 0], [0, 0]
                for index in indices_feature:
                    if ground_truth[index] == 1:
                        pos_total[0] += prediction[index]
                        pos_total[1] += 1
                    else:
                        neg_total[0] += prediction[index]
                        neg_total[1] += 1
                avg_pos = pos_total[0]/pos_total[1] if pos_total[1] != 0 else 0
                avg_neg = neg_total[0]/neg_total[1] if neg_total[1] != 0 else 0

                balance_data.append((feature_value, avg_pos, avg_neg))
            balance_data = torch.tensor(balance_data)
            outcome_per_feature.append((self.dataset.i2c[feature], balance_data))

        self.results = outcome_per_feature

    def _get_column(self, feature):
        return self.dataset.data[:,feature].squeeze(-1)

    def _get_ground_truth_column(self, params):
        ground_truth_column = params.get('ground_truth_column', '')
        assert ground_truth_column != '', 'Balance in class metrics needs a ground truth'
        return self.dataset.data[:,self.dataset.c2i[ground_truth_column]]

    def _get_prediction_column(self, params):
        prediction_column = params.get('prediction_column', '')
        assert prediction_column != '', 'Balance in class metrics needs a prediction'
        return self.dataset.data[:,self.dataset.c2i[prediction_column]]


    def show(self, raw_results=False) -> go.Figure:
        if raw_results:
            return self.results
        fig_list = []
        for feature, data in self.results:
            fig = go.Figure()

            if self.calc_pos:
                fig.add_trace(go.Bar(
                    name='Postive class',
                    x=data[:,0],
                    y=data[:,1],
                ))

            if self.calc_neg:
                fig.add_trace(go.Bar(
                    name='Negative class',
                    x=data[:,0],
                    y=data[:,2],
                ))

            fig.update_layout(
                title=f'Balance by Group for "{feature}"',
                xaxis_title='Group',
                yaxis_title='Average score',
                yaxis=dict(range=[0, 1]),
                bargap=0.2
            )

            fig_list.append(fig)
        return fig_list

print('loaded balance in class class')
