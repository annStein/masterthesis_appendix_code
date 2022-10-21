from enum import Enum

class FeatureSelector(Enum):
    FISHER_SCORE = "fisher_score"
    ANOVA = "anova"
    RELIEF_F = "relieff"
    RFE = "recursive feature elimination"

FilterFeatureSelector = [FeatureSelector.FISHER_SCORE, FeatureSelector.ANOVA, FeatureSelector.RELIEF_F]
WrapperFeatureSelector = [FeatureSelector.RFE]