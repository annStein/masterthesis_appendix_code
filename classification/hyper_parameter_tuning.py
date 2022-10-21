import logging
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from skopt.space import Integer, Real
from helper import build_label_and_feature_array, filter_features
from masterthesis_appendix_code.constants.definitions import (
    NUMBER_OF_FEATURES_TO_SELECT, LABEL_PATH, FEATURE_DIR, TRAIN_IDS, TEST_IDS, FEATURE_LIST_FILENAME, CLASSIFIER,
    CLASSIFIER_DIR
)
from classifier_config import config
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from infant_classifier import get_true_labels_per_infant, get_infant_prediction, get_infant_risk

# create training and test set
# import label data
labels = pd.read_csv(config[LABEL_PATH])
feat_dir = config[FEATURE_DIR]
config_clf : RandomForestClassifier = config[CLASSIFIER]
train_ids = config[TRAIN_IDS]
test_ids = config[TEST_IDS]
feat_file = os.path.join(config[CLASSIFIER_DIR], "feature_lists", f"{config[FEATURE_LIST_FILENAME]}.npy")
feature_number = config[NUMBER_OF_FEATURES_TO_SELECT]

train_labels, train_features, _ = build_label_and_feature_array(labels, feat_dir, train_ids)
test_labels, test_features, test_feature_ids = build_label_and_feature_array(labels, feat_dir, test_ids)
train_features, test_features = filter_features(train_features, test_features, feat_file, feature_number)

data = train_features.values
# define search space
params = dict()
params['n_estimators'] = Integer(100, 1000)
params['min_samples_leaf'] = Integer(20, 100)

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# define the search
search = BayesSearchCV(estimator=RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced"), search_spaces=params, n_jobs=-1, cv=cv,
                       scoring="f1_macro", random_state=42)
# perform the search
search.fit(train_features, train_labels)
# report the best result
best_params:dict = search.best_params_
print(best_params)

# put params in results dict
result_dict = {}
for par_key in best_params.keys():
    result_dict[par_key] = [best_params[par_key]]



clf = config_clf
clf: RandomForestClassifier = clf.set_params(**best_params)

print(clf.n_jobs)
print(clf.max_features)
print(clf.n_estimators)
print(clf.class_weight)
clf.fit(train_features, train_labels)
res = clf.predict(test_features)



result_dict["acc"] = [accuracy_score(test_labels, res)]
result_dict["f1"] = [f1_score(test_labels, res, average="macro")]
result_dict["oob"] = [1 - clf.oob_score_]
result_dict["sensitivity"] = [recall_score(test_labels, res)]
result_dict["specificity"] = [recall_score(test_labels, res, pos_label=0)]


tn, fp, fn, tp = confusion_matrix(test_labels, res).ravel()
print(f"TP: {tp}   FN: {fn}")
print(f"FP: {fp}   TN: {tn}")

infant_true_labels = np.fromiter(dict(sorted(get_infant_risk(test_ids, labels).items())).values(), dtype=bool)
true_pred_labels_per_infant = get_true_labels_per_infant(test_feature_ids, res)
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    infant_prediction = np.fromiter(
        dict(sorted(
            get_infant_prediction(true_pred_labels_per_infant, thresh).items()
        )).values(), dtype=bool
    )
    result_dict[f"infant_acc_{thresh}"] = accuracy_score(infant_true_labels, infant_prediction)
    result_dict[f"infant_f1_{thresh}"] = f1_score(infant_true_labels, infant_prediction, average="macro")
    result_dict[f"infant_sensitivity_{thresh}"] = recall_score(infant_true_labels, infant_prediction)
    result_dict[f"infant_specificity_{thresh}"] = recall_score(infant_true_labels, infant_prediction, pos_label=0)

filename = config[FEATURE_LIST_FILENAME]
pd.DataFrame(result_dict).to_csv(os.path.join(config[CLASSIFIER_DIR], "hyperparam_results", f"{filename}.csv"))