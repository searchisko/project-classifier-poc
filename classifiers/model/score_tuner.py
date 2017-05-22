import logging
import numpy as np
from scipy.optimize import minimize_scalar
import math

from sklearn.metrics import accuracy_score

import pandas as pd


def combined_accuracy(y_expected, y_actual, _):
    return accuracy_score(y_expected, y_actual)


def accuracy_for_category(y_expected, y_actual, label):
    label_expected = y_expected[y_expected == label]
    intersect = y_expected[np.where(y_expected == y_actual)]
    label_intersect = intersect[intersect == label]
    if len(label_expected) == 0:
        logging.warn("Accuracy of %s category evaluated on 0 samples" % label)
        return 1 if len(label_intersect) == 0 else 0
    else:
        return float(len(label_intersect)) / len(label_expected)


# A:0)
# expects non-indexed np arrays of expected and actual labels
def get_eval_groups_for_category(y_expected, cat_scores, category, threshold):
    cat_true_mask = y_expected == category
    true_docs_scores = cat_scores.loc[cat_true_mask]
    false_docs_scores = cat_scores.loc[~cat_true_mask]

    true_positives = true_docs_scores[true_docs_scores >= threshold]
    false_negatives = true_docs_scores[true_docs_scores < threshold]
    false_positives = false_docs_scores[false_docs_scores >= threshold]
    true_negatives = false_docs_scores[false_docs_scores < threshold]

    return {"TP": len(true_positives),
            "TN": len(true_negatives),
            "FP": len(false_positives),
            "FN": len(false_negatives)}


# A:1)
# computes precision and recall for given category by its TP/TN/FP/FN groups
# expects non-indexed np arrays of expected and actual labels
def precision_recall_for_category(y_expected, cat_scores, category, threshold):
    eval_groups = get_eval_groups_for_category(y_expected, cat_scores, category, threshold)
    precision = eval_groups["TP"] / (eval_groups["TP"] + eval_groups["FP"])
    recall = eval_groups["TP"] / (eval_groups["TP"] + eval_groups["FN"])

    return precision, recall


# B)
# evaluates the f-score for given f-beta by tuning the threshold param separating TP/FN and TN/FP on category scores
def f_score_for_category(y_expected, cat_scores, category, threshold, beta):
    precision, recall = precision_recall_for_category(y_expected, cat_scores, category, threshold)
    return (1 + (beta ^ 2)) * ((precision * recall) / precision + recall)


# C)
# maximize the f-score for given f-beta by tuning the threshold param separating TP/FN and TN/FP on category scores
def maximize_f_score(y_expected, cat_scores, category, beta):
    def inv_f_score_func_wrapper(threshold):
        return -f_score_for_category(y_expected, cat_scores, category, threshold, beta)

    opt_threshold = minimize_scalar(fun=inv_f_score_func_wrapper, method="bounded", bounds=(0, 1)).threshold

    return opt_threshold


# D:1)
# series of probs, scalar of original_cat_threshold
def normalize_probs(cat_probs, original_cat_threshold):
    # relative ratios of spaces divided by a split of <0, 1> prob space by a value of doc prob
    # where space is determined by a distance of doc_i prob from original_cat_threshold
    target_threshold = 0.5
    # the figure projects points into <0, 1> only with symmetric target_threshold = 0.5

    return target_threshold * (cat_probs - original_cat_threshold) + target_threshold


# D:2)
# normalizes the category scores so that the relevance of documents is easily evaluated using the unified threshold
# to distinguish non-relevant from relevant
# sensitivity (=precision/recall ratio) of a search can be customized by parametrized f-score for category
def normalize_cat_scores(y_expected, cat_scores, category, beta):
    #
    opt_cat_threshold = maximize_f_score(y_expected, cat_scores, category, beta)
    # moves the scores so that the threshold gets to 0.5, and ideally all scores are scaled in <0, 1>
    norm_cat_scores = normalize_probs(cat_scores, opt_cat_threshold)
    return norm_cat_scores


# F)
# computes beta that is empirically presumed to convey the expectations of the search engine for given categories
# we want to weight beta as log-dependent to categories sizes
# this way we'll prefer a weight of precision for bigger categories and recall for categories with little content
def beta_for_categories_provider(y_expected):
    # function retrieves beta in interval <0.2, 5>
    # params were set to log-approximate the interval for categories between sizes of <100, 20 000> content
    def beta_func(cat_size):
        a = 0.905948
        b = 9.17204
        return -a*math.log(cat_size)+b

    cats_sizes = y_expected.value_counts()
    cats_betas = cats_sizes.apply(beta_func)

    return cats_betas


# D:3)
# gets df of scores of some content for all categories and expected labels
# will tune the scores of categories so that the same threshold for all categories can be applied
# and still the selected content will qualitatively persist the maximized F-score
def normalize_all_cats_scores(y_expected, scores_df):
    norm_scores_df = pd.DataFrame()
    categories_betas = beta_for_categories_provider(y_expected)
    for cat_label in scores_df.keys().unique():
        norm_scores_df[cat_label] = normalize_cat_scores(y_expected, scores_df[cat_label],
                                                         cat_label, categories_betas[cat_label])
    return norm_scores_df


# E) TODO
def evaluate(expected_labels, actual_labels):
    pass
