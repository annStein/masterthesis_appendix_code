import datetime

FEATURE_DIR = "feature_dir"
LABEL_PATH = "label_path"
CLASSIFIER = "classifier"
NUMBER_OF_FEATURES_TO_SELECT = "number_of_features"
FEATURE_SELECTOR = "feature_selector"
NUMBER_OF_REPETITIONS = "number_of_repetitions"
FEAT_CORR_REMOVAL = "feat_correlation_removal"
TRAIN_IDS = "train_ids"
TEST_IDS = "test_ids"
FEATURE_LIST_FILENAME = "feature_list_filename"
RESULT_FILENAME = "result_filename"
CLASSIFIER_DIR = "classifier_dir"
# ---------------------------------------------------------------------
# Landmark Constants
UPPER_BODY_PARTS = ["Nose", "Eye_inner_left", "Eye_left", "Eye_outer_left", "Eye_inner_right",
                                    "Eye_right", "Eye_outer_right",
                                    "Ear_left", "Ear_right", "Mouth_left", "Mouth_right", "Shoulder_left",
                                    "Shoulder_right",
                                    "Elbow_left", "Elbow_right", "Wrist_left", "Wrist_right", "Pinky_left",
                                    "Pinky_right", "Index_left",
                                    "Index_right", "Thumb_left", "Thumb_right"]
LOWER_BODY_PARTS = ["Hip_left", "Hip_right", "Knee_left", "Knee_right", "Ankle_left",
                    "Ankle_right", "Heel_left", "Heel_right", "Foot_index_left", "Foot_index_right"]
BODY_PARTS = UPPER_BODY_PARTS + LOWER_BODY_PARTS
# from literature (3.3m/s = 110mm per frame when framerate 30fps)
CALIBRATION_JUMP_THRESHOLD = 110
# Threshold to identify framedrops
FRAMERATE_JUMP_THESHOLD = 0.1
# ---------------------------------------------------------------------
# App Constants
RUN_TIMESTAMP = datetime.datetime.now()