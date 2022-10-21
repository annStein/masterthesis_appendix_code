import numpy as np
import pandas as pd
import os
import re

def _find_2nd_(string, substring):
   return string.find(substring, string.find(substring) + 1)


def _find_3rd_(string, substring):
   return string.find(substring, string.find(substring, string.find(substring) + 1) + 1)


def _get_id_(filename):
    underscore_before_id = _find_2nd_(filename, "_")
    underscore_after_id = _find_3rd_(filename, "_")
    # filter only digits (in case of id being 046a)
    id = re.sub('\D', '', filename[underscore_before_id + 1:underscore_after_id])
    return int(id)

def _get_feature_filename_without_id_(filename):
    return filename[_find_3rd_(filename, "_"):]


def build_label_and_feature_array(label_df, feature_dir, ids_for_features):
    # number of feature files refers to the number of different feature csv files per ID
    labels = []
    ids_per_label = []

    feature_dict = {}
    for root, dirs, files in os.walk(feature_dir):
        for filename in files:
            id = _get_id_(filename)
            if id in ids_for_features:
                feature_df = pd.read_csv(os.path.join(root, filename))
                feature_filename = _get_feature_filename_without_id_(filename)
                if feature_filename in feature_dict.keys():
                    feature_dict[feature_filename] = pd.concat([feature_dict[feature_filename], feature_df], axis=0).reset_index(drop=True)
                else:
                    feature_dict[feature_filename] = feature_df

                if id not in ids_per_label:
                    # create label for every chunk
                    label = label_df.loc[label_df['ID'] == id].iloc[0]["risk"]
                    number_of_chunks = len(feature_df)
                    labels = labels + ([label] * number_of_chunks)
                    ids_per_label = ids_per_label + ([id] * number_of_chunks)

    features = pd.concat(feature_dict.values(), axis=1)
    # remove NaN values
    nan_indices = features.isnull().any(1).to_numpy().nonzero()[0]
    for idx in reversed(nan_indices):
        del labels[idx]
        del ids_per_label[idx]
    features = features.dropna()

    return np.array(labels), features, np.array(ids_per_label)


def filter_features(train_features, test_features, feature_list_filename, number_of_features = None):
    cols = np.load(
        feature_list_filename,
        allow_pickle=True
    )
    if number_of_features != None:
        cols = cols[:number_of_features]
    filtered_train = train_features.loc[:, cols]
    filtered_test = test_features.loc[:, cols]
    return filtered_train, filtered_test
