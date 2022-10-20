import os

from sklearn.ensemble import RandomForestClassifier

from masterthesis_appendix_code.constants.defintions import (
    FEATURE_DIR, LABEL_PATH, NUMBER_OF_FEATURES_TO_SELECT, CLASSIFIER, TRAIN_IDS, TEST_IDS,
    FEATURE_LIST_FILENAME, CLASSIFIER_DIR
)
config = {
    # dir of this config
    CLASSIFIER_DIR: os.path.dirname(os.path.abspath(__file__)),
    # ---------------------------------------------------------------------
    # Features are loaded from this directory (and any subdirectories)
    FEATURE_DIR: r"features",
    # ---------------------------------------------------------------------
    LABEL_PATH: r"sinda_label.csv",
    # ---------------------------------------------------------------------
    # is only used if a filter feature selector is used
    NUMBER_OF_FEATURES_TO_SELECT: None,
    # ---------------------------------------------------------------------
    # classifier which should be used
    CLASSIFIER:RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=42, class_weight="balanced"),
    # ---------------------------------------------------------------------
    # ranked list of features
    FEATURE_LIST_FILENAME: "all_sinda_rfe",
    # ---------------------------------------------------------------------
    # IDs for training and set
    TRAIN_IDS: [33, 60, 40, 68, 67, 37, 36, 63, 58, 35, 59, 53, 32, 28, 57, 65, 62, 45, 46, 55, 49],
    TEST_IDS:  [34, 48, 52, 66, 39, 41, 44, 47, 50, 54, 61]
}
