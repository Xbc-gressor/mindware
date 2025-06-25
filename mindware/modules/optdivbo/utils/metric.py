from math import sqrt
import numpy as np
import pandas as pd


def diversity(pred1, pred2):
    return np.mean(np.linalg.norm(pred1 - pred2, ord=2, axis=1)) / sqrt(2)



def opt_diversity(pred1, pred2, score_name, y_true=None):
    if score_name == 'cross_entropy':
        n = y_true.shape[0]
        rows = np.arange(n) 

        y_true = y_true.astype(int)

        pred1 = pred1[rows, y_true]
        pred2 = pred2[rows, y_true]

        epsilon = 1e-10 
        sum_pred2 = np.sum(pred2)
        safe_sum = sum_pred2 if sum_pred2 > epsilon else epsilon

        log_terms = np.log(np.maximum(pred1, epsilon) / np.maximum(safe_sum, epsilon))

        div = (1/n) * np.sum(log_terms) + np.log(n)
        return div
    
    elif score_name == 'mse':
        n = y_true.shape[0]
        gap1 = y_true - pred1
        gap2 = y_true - pred2
        i, j = np.triu_indices(n, k=1)
        div = 2/n **2 * np.sum(gap1[i] * gap2[j])

        return div
    elif score_name == 'mae':
        n = y_true.shape[0]
        gap1 = y_true - pred1
        gap2 = y_true - pred2
        i, j = np.triu_indices(n, k=1)

        div = 1 /n * np.sum(np.abs(gap1)) + 2**0.5 / n *np.sum(np.abs(gap1[i] * gap2[j]) ** 0.5)
        return div
    else:
        return np.mean(np.linalg.norm(pred1 - pred2, ord=2, axis=1)) / sqrt(2)

def calculate_ranking(score_dict, ascending=False):
    rank_dict = dict()
    for key in score_dict:
        value_list = pd.Series(list(score_dict[key]))
        rank_array = np.array(value_list.rank(ascending=ascending))
        rank_dict[key] = rank_array

    return rank_dict
