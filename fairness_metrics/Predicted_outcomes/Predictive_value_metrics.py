import torch
import itertools
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


        prediction = self._get_prediction_column(params)
        prediction = (prediction > 0.8).float().squeeze()
        ground_truth = self._get_ground_truth_column(params)
        #Currently 80% certainty of model indicates a 1 prediction but can be changed
        
        protected_values = params.get('protected_values', torch.zeros(len(dataset.i2c), dtype=bool))
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
            unique_values = torch.unique(protected_col)

            PPV_dict = {}
            NPV_dict = {}
            for val in unique_values:
                mask = protected_col == val
                preds = prediction[mask]
                truth = ground_truth[mask]
                if len(preds) > 0:
                    TP = ((preds == 1) & (truth == 1)).sum()
                    TN = ((preds == 0) & (truth == 0)).sum()
                    FP = ((preds == 1) & (truth == 0)).sum()
                    FN = ((preds == 0) & (truth == 1)).sum()

                    ppv = TP / (TP + FP)
                    npv = TN / (TN + FN)
                else:
                    ppv = float('nan')
                    npv = float('nan')
                PPV_dict[int(val)] = ppv
                NPV_dict[int(val)] = npv

            self.results[attr] = {
                "ppv": PPV_dict,
                "npv": NPV_dict
            }

    def _get_column(self, feature):
        return self.dataset.data[:,feature].squeeze(-1)

    def _get_ground_truth_column(self, params):
        ground_truth_column = params.get('ground_truth_column', '')
        assert ground_truth_column != '', 'This metric needs a ground truth'
        return self.dataset.data[:,self.dataset.c2i[ground_truth_column]]

    def _get_prediction_column(self, params):
        prediction_column = params.get('prediction_column', '')
        assert prediction_column != '', 'This metric needs a prediction'
        return self.dataset.data[:,self.dataset.c2i[prediction_column]]
    
    def show(self, raw_results=False):
        if raw_results:
            return self.results
        for attr_name in self.results:
            attr_data = self.results[attr_name]
            ppv = attr_data['ppv']
            npv = attr_data['npv']
            
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='PPV',
                x=[str(k) for k in ppv.keys()],
                y=list(ppv.values())
            ))

            fig.add_trace(go.Bar(
                name='NPV',
                x=[str(k) for k in npv.keys()],
                y=list(npv.values())
            ))

            fig.update_layout(
                title=f'PPVs and NPVs by Group for "{attr_name}"',
                xaxis_title='Group',
                yaxis_title='PPVs and NPVs',
                yaxis=dict(range=[0, 1]),
                bargap=0.2
            )

            fig.show()