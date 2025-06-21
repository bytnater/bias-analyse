'''
This file includes a class for the fairness metrics 'balance in positve/negative class'

author: Mick
'''

import torch
import pandas as pd
import matplotlib.pyplot as plt

#################################################################
### for testing purpuse
#################################################################

DATA_PATH_SYNTH = 'data/synth_data.csv'
SAVED_PRESET_PATH = '/home/itsmick_/Documents/UvA/Tweedejaarsproject/project/bias-analyse/checkpoints/presets/'

def load_csv_to_torch(path=DATA_PATH_SYNTH):
    df = pd.read_csv(path)
    i2c = df.columns.tolist()
    c2i = {c: i for i, c in enumerate(i2c)}
    data_torch = torch.from_numpy(df.values)
    return data_torch, (i2c, c2i)

class Dataset():
    def __init__(self, PATH):
        data, (i2c, c2i) = load_csv_to_torch(PATH)
        self.data = data
        self.i2c = i2c
        self.c2i = c2i

dataset = Dataset('/home/itsmick_/Documents/UvA/Tweedejaarsproject/project/bias-analyse/data/altered_data/data_pred_ground_altered_pred_biased.csv')
params = torch.load(SAVED_PRESET_PATH + 'preset1.pt')

#################################################################

class balance_in_pos_neg:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.calc_pos = params.get('balance_pos', True)
        self.calc_neg = params.get('balance_neg', True)
        assert self.calc_pos + self.calc_neg != 0, 'Select at least one metric'

        self.ground_truth_column = params.get('ground_truth_column', '')
        self.prediction_column = params.get('prediction_column', '')

        assert self.ground_truth_column != '', 'This metric needs a ground truth'
        assert self.prediction_column != '', 'This metric needs a prediction'

        self.check_features = params.get('protected_values', torch.zeros(len(self.dataset.i2c), dtype=bool))
        indices = self.check_features.nonzero()
        # self.dataset.data[:,params['protected_values']]  ## filter all unimportent


        # calculation
        outcome_per_feature: list[tuple[str, list[int, float]]] = []
        for feature in indices:
            # print('feature i atm:',feature)
            feature_column = self.dataset.data[:,feature].squeeze(-1)
            ground_truth = self.dataset.data[:,self.dataset.c2i[self.ground_truth_column]]
            prediction = self.dataset.data[:,self.dataset.c2i[self.prediction_column]]
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
                # print(avg_pos)
                # print(avg_neg)

                balance_data.append((feature_value, avg_pos, avg_neg))
            balance_data = torch.tensor(balance_data)
            outcome_per_feature.append((self.dataset.i2c[feature], balance_data))

        self.results = outcome_per_feature

    def show(self, raw_results=False):
        if raw_results:
            return self.results
        for feature, data in self.results:
            if self.calc_pos:
                plt.title(f'Postive balance: "{feature}"')
                plt.bar(data[:,0], data[:,1])
                plt.show()
            if self.calc_neg:
                plt.title(f'Negative balance: "{feature}"')
                plt.bar(data[:,0], data[:,2])
                plt.show()

#################################################################
### for testing purpuse
#################################################################
metric = balance_in_pos_neg(dataset, params)
metric.show()