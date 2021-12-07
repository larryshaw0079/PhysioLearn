"""
@Time    : 2021/9/22 11:30
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score


def report_performance(scores: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(scores, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    roc = roc_auc_score(labels, scores)
    map = average_precision_score(labels, scores)

    cm = confusion_matrix(labels, predictions)
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

    performance = {'accuracy': [accuracy],
                   **{f'accuracy_class_{i}': [acc] for i, acc in enumerate(accuracy_per_class.tolist())},
                   'f1_micro': [f1_micro], 'f1_macro': [f1_macro], 'roc': [roc], 'map': [map]}

    return pd.DataFrame(performance).transpose()
