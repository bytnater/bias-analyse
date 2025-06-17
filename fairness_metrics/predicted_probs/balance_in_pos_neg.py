'''
This file includes a class for the fairness metrics 'balance in positve/negative class'

author: Mick
'''

class balance_in_pos_neg:
    def __init__(self, dataset, calc_pos = False, calc_neg = False):
        self.calc_pos = calc_pos
        self.calc_neg = calc_neg
        print('done')

    def print(self):
        print(self.calc_neg, self.calc_pos)