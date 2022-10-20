import numpy as np
import pandas as pd


def get_true_labels_per_infant(test_id_per_label, results):
    """"
    returns percentage of true labels per infant
    """
    assert len(test_id_per_label) == len(results)
    id_label_arr = np.vstack((test_id_per_label, results)).transpose()
    result = {}

    for infant_id in np.unique(test_id_per_label):
        id_labels = [chunk[1] for chunk in id_label_arr if chunk[0] == infant_id]
        number_of_true_labels = np.count_nonzero(id_labels)
        result[infant_id] = number_of_true_labels / np.size(id_labels)
    # sort by infant id
    return dict(sorted(result.items()))

def get_infant_prediction(infant_prediction: dict, threshold = 0.5):
    """"
    @param infant_prediction: result of get_true_labels_per_infant
    """
    result = {}
    for infant_id in infant_prediction.keys():
        result[infant_id] = infant_prediction[infant_id] >= threshold
    # sort by infant id
    return dict(sorted(result.items()))

def get_infant_risk(infant_ids, labels):
    result = {}
    for infant_id in infant_ids:
        result[infant_id] = labels.loc[labels['ID'] == infant_id, 'risk'].values[0]
    return result
