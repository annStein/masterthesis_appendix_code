from sklearn.model_selection import train_test_split

from feat_sel_config import config
import pandas as pd
from masterthesis_appendix_code.constants.definitions import (
    FEATURE_SELECTOR, LABEL_PATH, FEATURE_DIR, TRAIN_IDS, CLASSIFIER
)
from helper import  build_label_and_feature_array
from feature_selection_methods import select_features

def run_feat_sel():
    feat_selector = config[FEATURE_SELECTOR]
    labels = pd.read_csv(config[LABEL_PATH])
    feat_dir = config[FEATURE_DIR]
    train_ids = config[TRAIN_IDS]
    estimator = config[CLASSIFIER]

    train_labels, train_features, _ = build_label_and_feature_array(labels, feat_dir, train_ids)

    # build test and validation set
    train_features,test_features,train_labels,test_labels = train_test_split(train_features, train_labels, test_size=0.1, random_state=42, stratify=train_labels)
    print(train_features.head())
    train_features, test_features = \
        select_features(
            train_features, train_labels, train_features, train_features.shape[1], feat_selector, estimator
        )
