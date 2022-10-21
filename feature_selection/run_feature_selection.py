
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from feat_sel_config import config
from masterthesis_appendix_code.constants.definitions import (
    FEATURE_DIR, LABEL_PATH, FEATURE_SELECTOR, CLASSIFIER,
    TRAIN_IDS, TEST_IDS, RESULT_FILENAME, CLASSIFIER_DIR
)
from helper import build_label_and_feature_array, filter_features
from selection_methods import FilterFeatureSelector, FeatureSelector, \
    WrapperFeatureSelector
from feature_selection_methods import select_features, select_features_RFE
import feature_selection


def run_filter():
    feature_selection.run_feat_sel()
    # import label data
    labels = pd.read_csv(config[LABEL_PATH])
    feat_dir = config[FEATURE_DIR]
    config_clf : RandomForestClassifier = config[CLASSIFIER]
    train_ids = config[TRAIN_IDS]

    result_filename = config[RESULT_FILENAME]
    feat_file = os.path.join(config[CLASSIFIER_DIR], "feature_lists", f"{config[FEATURE_SELECTOR].value}.npy")

    train_labels, train_features, _ = build_label_and_feature_array(labels, feat_dir, train_ids)
    train_features, validation_features, train_labels, validation_labels = train_test_split(train_features, train_labels,
                                                                                test_size=0.1, random_state=42,
                                                                                stratify=train_labels)
    max_features = train_features.shape[1]

    result_dict = {"number_of_features": [], "acc": [], "f1": [], "oob": [], "sensitivity": [], "specificity": []}

    for n in range(5, max_features, 5):
        filtered_train_features, filtered_test_features = filter_features(train_features, validation_features, feat_file, n)

        # train classifier and predict results for test data
        clf = config_clf
        clf.fit(filtered_train_features, train_labels)
        results = clf.predict(filtered_test_features)


        result_dict["number_of_features"].append(n)
        result_dict["acc"].append(accuracy_score(validation_labels, results))
        result_dict["f1"].append(f1_score(validation_labels, results, average="macro"))
        result_dict["oob"].append(1 - clf.oob_score_)
        result_dict["sensitivity"].append(recall_score(validation_labels, results))
        result_dict["specificity"].append(recall_score(validation_labels, results, pos_label=0))

    # quick and dirty... add another run for all features
    filtered_train_features, filtered_test_features = filter_features(train_features, validation_features, feat_file, max_features)

    # train classifier and predict results for test data
    clf = config_clf
    clf.fit(filtered_train_features, train_labels)
    results = clf.predict(filtered_test_features)

    result_dict["number_of_features"].append(max_features)
    result_dict["acc"].append(accuracy_score(validation_labels, results))
    result_dict["f1"].append(f1_score(validation_labels, results, average="macro"))
    result_dict["oob"].append(1 - clf.oob_score_)
    result_dict["sensitivity"].append(recall_score(validation_labels, results))
    result_dict["specificity"].append(recall_score(validation_labels, results, pos_label=0))

    filename = result_filename
    pd.DataFrame(result_dict).to_csv(os.path.join(config[CLASSIFIER_DIR], "feature_selection_results", f"{filename}.csv"))

def run_rfe(clf, result_filename):
    # import label data
    labels = pd.read_csv(config[LABEL_PATH])
    feat_dir = config[FEATURE_DIR]
    config_clf : RandomForestClassifier = config[CLASSIFIER]
    train_ids = config[TRAIN_IDS]
    test_ids = config[TEST_IDS]

    train_labels, train_features, _ = build_label_and_feature_array(labels, feat_dir, train_ids)
    train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels,
                                                                                test_size=0.1, random_state=42, stratify=train_labels)

    result_dict = {"number_of_features": [], "acc": [], "f1": [], "oob": [], "sensitivity": [], "specificity": []}

    # quick and dirty...
    filtered_train_features, filtered_test_features, classifier = \
        select_features_RFE(
            train_features, train_labels, test_features, clf
        )

    # train classifier and predict results for test data
    clf.fit(filtered_train_features, train_labels)
    results = clf.predict(filtered_test_features)

    result_dict["number_of_features"].append(filtered_train_features.shape[1])
    result_dict["acc"].append(accuracy_score(test_labels, results))
    result_dict["f1"].append(f1_score(test_labels, results, average="macro"))
    result_dict["oob"].append(1 - clf.oob_score_)
    result_dict["sensitivity"].append(recall_score(test_labels, results))
    result_dict["specificity"].append(recall_score(test_labels, results, pos_label=0))

    filename = result_filename
    pd.DataFrame(result_dict).to_csv(os.path.join(config[CLASSIFIER_DIR], "feature_selection_results", f"{filename}.csv"))


def select_features_with_method(feat_selector, test_features, train_features, train_labels, n_feat_select):
    if feat_selector in FeatureSelector:
        if feat_selector in FilterFeatureSelector or feat_selector in WrapperFeatureSelector:
            train_features, test_features = \
                select_features(
                    train_features, train_labels, test_features, n_feat_select, feat_selector
                )
        else:
            raise Exception("Feature selection method is not found.")
    return test_features, train_features


def filter_features_with_file(feature_columns_filename, training_features, test_features):
    feature_columns = np.load(feature_columns_filename)
    filtered_training_features = training_features[feature_columns]
    filtered_test_features = test_features[feature_columns]
    return filtered_training_features, filtered_test_features


if config[FEATURE_SELECTOR] == FeatureSelector.RFE:
    run_rfe(config[CLASSIFIER], config[RESULT_FILENAME])
else:
    run_filter(config[CLASSIFIER], None)
