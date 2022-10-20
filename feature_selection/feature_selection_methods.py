import os

import pandas as pd
import numpy as np
from ReliefF import ReliefF
from matplotlib import pyplot as plt
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.model_selection import StratifiedKFold
from masterthesis_appendix_code.constants.defintions import (
    CLASSIFIER_DIR
)
from feat_sel_config import config

from masterthesis_appendix_code.constants.selection_methods import FeatureSelector


def select_features(
        train_features: pd.DataFrame,
        labels: np.array,
        test_features: pd.DataFrame,
        n: int,
        method: FeatureSelector,
        estimator: RandomForestClassifier = None,
) -> (pd.DataFrame,pd.DataFrame):
    if method == FeatureSelector.FISHER_SCORE:
        return select_features_fisher_score(train_features, labels, test_features, n)
    elif method == FeatureSelector.ANOVA:
        return select_features_ANOVA_F_value(train_features, labels, test_features, n)
    elif method == FeatureSelector.RELIEF_F:
        return select_features_relieff(train_features, labels, test_features, n)
    elif method == FeatureSelector.RFE:
        return select_features_RFE(train_features, labels, test_features, estimator)
    else:
        raise Exception("Method for Feature Selection not found.")


def select_features_relieff(
        train_features: pd.DataFrame,
        labels: np.array,
        test_features: pd.DataFrame,
        n: int
) -> (pd.DataFrame,pd.DataFrame):
    # configure to select all features
    fs = ReliefF(n_features_to_keep=n)
    # learn relationship from training data
    filtered_train_features = fs.fit_transform(train_features.to_numpy(), labels)
    # transform train input data
    # filtered_train_features = fs.transform(train_features.to_numpy())
    # transform test input data
    ranks = fs.top_features
    cols_with_rank = pd.Series(ranks, train_features.columns)
    n_cols_with_highest_rank = cols_with_rank.nlargest(n).keys().to_numpy()
    np.save(os.path.join(config[CLASSIFIER_DIR], "feature_lists", "relieff.npy"), n_cols_with_highest_rank)
    filtered_test_features = fs.transform(test_features.to_numpy())
    return filtered_train_features, filtered_test_features


def select_features_fisher_score(
        train_features: pd.DataFrame,
        labels: np.array,
        test_features: pd.DataFrame,
        n: int
) -> (pd.DataFrame,pd.DataFrame):
    ranks = fisher_score.fisher_score(train_features.to_numpy(), labels)
    cols_with_rank = pd.Series(ranks, train_features.columns)
    n_cols_with_highest_rank = cols_with_rank.nlargest(n).keys().to_numpy()
    np.save(os.path.join(config[CLASSIFIER_DIR], "feature_lists", "fisher_score.npy"), n_cols_with_highest_rank)
    filtered_train_features = train_features.filter(n_cols_with_highest_rank)
    filtered_test_features = test_features.filter(n_cols_with_highest_rank)
    return filtered_train_features, filtered_test_features


def select_features_ANOVA_F_value(
        train_features: pd.DataFrame,
        labels: np.array,
        test_features: pd.DataFrame,
        n: int
) -> (pd.DataFrame,pd.DataFrame):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k=n)
    # learn relationship from training data
    fit = fs.fit(train_features, labels)
    # transform train input data
    filtered_train_features = fs.transform(train_features)
    # transform test input data
    filtered_test_features = fs.transform(test_features)
    cols = fs.get_support(indices=True)
    feat_col_names = train_features.iloc[:, cols].columns
    ranked_features= pd.DataFrame({
        "Features": pd.Series(feat_col_names, dtype=str),
        "Scores": pd.Series(fs.scores_, dtype=float)
    }).sort_values(by=["Scores"], ascending=False)["Features"].to_numpy()
    np.save(os.path.join(config[CLASSIFIER_DIR], "feature_lists", "anova.npy"), ranked_features)
    return filtered_train_features, filtered_test_features


def select_features_RFE(
        train_features: pd.DataFrame,
        labels: np.array,
        test_features: pd.DataFrame,
        estimator
) -> (pd.DataFrame,pd.DataFrame):
    # configure to select all features
    fs = RFECV(estimator=estimator, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro")
    # learn relationship from training data
    fs.fit(train_features, labels)
    # transform train input data
    filtered_train_features = fs.transform(train_features)
    # transform test input data
    filtered_test_features = fs.transform(test_features)
    print(f"features before: {len(train_features.columns)}     features after: {len(filtered_train_features[0])}")
    print(f"selected features: {fs.get_feature_names_out()}")
    np.save(os.path.join(config[CLASSIFIER_DIR], "feature_lists","rfe.npy"), fs.get_feature_names_out())
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(0, len(fs.cv_results_["mean_test_score"])),
        fs.cv_results_["mean_test_score"],
    )
    plt.show()
    return filtered_train_features, filtered_test_features, estimator


def remove_features_with_high_correlation(train_features: pd.DataFrame, test_features: pd.DataFrame):
    corr_m = train_features.corr().abs()
    # get only elements about diagonal
    upper_tri = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(np.bool))
    # drop columns with high correlation
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    filtered_train_feat = train_features.drop(to_drop, axis=1)
    filtered_test_feat = test_features.drop(to_drop, axis=1)
    return filtered_train_feat, filtered_test_feat
