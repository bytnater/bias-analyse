'''
This file includes a class for the fairness metrics 'test-fairness and well-calibration'

author: Mick
'''

import torch
import numpy as np
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

class well_calibration:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.params = params

        self.ground_truth_column = params.get('ground_truth_column', '')
        self.prediction_column = params.get('prediction_column', '')

        assert self.ground_truth_column != '', 'This metric needs a ground truth'
        assert self.prediction_column != '', 'This metric needs a prediction'

        check_features = params.get('protected_values', torch.zeros(len(self.dataset.i2c), dtype=bool))
        indices = check_features.nonzero()


        # calculation
        outcome_per_feature = []
        for feature in indices:
            feature_column = self.dataset.data[:,feature].squeeze(-1)
            ground_truth = self.dataset.data[:,self.dataset.c2i[self.ground_truth_column]]
            prediction = self.dataset.data[:,self.dataset.c2i[self.prediction_column]]
            prediction = torch.round(prediction / 0.1)  ## bins the into bins from 0 to 10
            unique_feature_values = feature_column.unique()
            # print('unique values', unique_feature_values)

            calibration_data = []
            for feature_value in unique_feature_values:

                indices_feature = (feature_column == feature_value).nonzero().squeeze(-1)
                bins = np.arange(0,10.5,1)
                data = {key: [0, 0] for key in bins}

                for index in indices_feature:
                    key = int(prediction[index])
                    data[key][1] += 1
                    if ground_truth[index] == 1:
                        data[key][0] += 1

                refit_data = [feature_value.item()]
                for data_key, data_item in data.items():
                    fraction = data_item[0]/data_item[1] if data_item[1] else 0
                    refit_data.append(fraction)

                calibration_data.append(refit_data)
            calibration_data = torch.tensor(calibration_data)
            outcome_per_feature.append((self.dataset.i2c[feature], calibration_data))
        
        self.results = outcome_per_feature

    def show(self, raw_results=False):
        if raw_results:
            return self.results
        bins = np.arange(0,10.5,1)
        for feature, data in self.results:
                plt.title(f'Fairness scale: "{feature}"')
                if len(data) > 5:
                    bin_ranges = (np.linspace(0,len(data)-1,4)+.5).astype(int)
                    for i, ni in zip(bin_ranges[:-1], bin_ranges[1:]):
                        new_data = sum(data[i:ni,1:])/len(data[i:ni,1:])
                        plt.plot(bins, new_data, label=[f'from {data[i,0]} to {data[ni,0]}'])

                else:
                    for group_data in data:
                        plt.plot(bins, group_data[1:], label=[f'{group_data[0]}'])
                
                plt.legend()
                plt.show()


#################################################################
### for testing purpuse
#################################################################
metric = well_calibration(dataset, params)
metric.show()