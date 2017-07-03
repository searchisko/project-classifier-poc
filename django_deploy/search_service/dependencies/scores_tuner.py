import cPickle
import logging
import numpy as np
from scipy.optimize import minimize_scalar

import pandas as pd

# TODO: set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)

general_search_threshold = 0.5

"""
Stateful provider of scaling service for scores of categories as inferred by classifiers
"""


class ScoreTuner:
    cats_original_thresholds = pd.Series()
    trained = False

    def __init__(self):
        pass

    """
    Tunes the optimal score threshold for each category according to cats_scores Data Frame
    and predefined categories betas (respecting the precision/recall ratio desired for categories according to its size)
    to be used for tuning the score of a new content
    """
    def train_categories_thresholds(self, y_expected, cats_scores):
        cats_betas = self.beta_for_categories_provider(y_expected)
        categories_labels = pd.Series(data=cats_betas.index, index=cats_betas.index)

        self.cats_original_thresholds = categories_labels.apply(
            lambda cat_label: self.maximize_f_score(y_expected, cats_scores[cat_label], cat_label, cats_betas.loc[cat_label]))

        logging.info("Original categories thresholds tuned to:\n%s" % self.cats_original_thresholds)
        self.trained = True

    """
    Tunes the scores of a new data frame of content according to the optimized thresholds
    learnt in tune_categories_thresholds.
    Bulk operation.
    """
    def tune_new_docs_scores(self, docs_scores):
        norm_scores_df = pd.DataFrame(columns=docs_scores.keys(), index=docs_scores.index)

        for cat_label in docs_scores.keys():
            norm_scores_df[cat_label] = self.tune_cat_scores(docs_scores[cat_label], cat_label)
        return norm_scores_df

    """
    Tunes the scores of a single document.
    Expects doc_scores as series with category labels as index
    """
    def tune_new_doc_scores(self, doc_scores_series):
        doc_scores_df = pd.DataFrame(data=doc_scores_series.values, columns=doc_scores_series.index)
        doc_scores_df_tuned = self.tune_new_docs_scores(doc_scores_df)
        return pd.Series(data=doc_scores_df_tuned, index=doc_scores_df_tuned.columns)

    # A:0)
    # expects non-indexed np arrays of expected and actual labels
    @staticmethod
    def get_eval_groups_for_category(y_expected, cat_scores, category, threshold):
        cat_true_mask = y_expected == category
        true_docs_scores = cat_scores.loc[cat_true_mask]
        false_docs_scores = cat_scores.loc[~cat_true_mask]

        true_positives = true_docs_scores[true_docs_scores >= threshold]
        false_negatives = true_docs_scores[true_docs_scores < threshold]
        false_positives = false_docs_scores[false_docs_scores >= threshold]
        true_negatives = false_docs_scores[false_docs_scores < threshold]

        ratios = {"TP": len(true_positives),
                  "TN": len(true_negatives),
                  "FP": len(false_positives),
                  "FN": len(false_negatives)}
        # logging.info("get_eval_groups_for_category on %s separated by threshold %s returns: %s"
        #   % (category, threshold, ratios))

        return ratios

    # A:1)
    # computes precision and recall for given category by its TP/TN/FP/FN groups
    # expects non-indexed np arrays of expected and actual labels
    def precision_recall_for_category(self, y_expected, cat_scores, category, threshold):
        eval_groups = self.get_eval_groups_for_category(y_expected, cat_scores, category, threshold)
        precision = float(eval_groups["TP"]) / (eval_groups["TP"] + eval_groups["FP"]) \
            if eval_groups["TP"] + eval_groups["FP"] > 0 else 0
        recall = float(eval_groups["TP"]) / (eval_groups["TP"] + eval_groups["FN"]) \
            if eval_groups["TP"] + eval_groups["FN"] > 0 else 0

        # logging.info("precision_recall_for_category returns: %s, %s" % (precision, recall))

        return precision, recall

    # B)
    # evaluates the f-score for given f-beta by tuning the threshold param separating TP/FN and TN/FP on category scores
    def f_score_for_category(self, y_expected, cat_scores, category, threshold, beta):
        precision, recall = self.precision_recall_for_category(y_expected, cat_scores, category, threshold)
        f_score = (beta * beta + 1) * precision * recall / (
        beta * beta * precision + recall) if precision + recall > 0 else 0

        # logging.info("f_score_for_category params: cat %s, beta %s, threshold %s -> fscore: %s"
        #              % (category, beta, threshold, f_score))
        return f_score

    # C)
    # maximize the f-score for given f-beta by tuning the threshold param separating TP/FN and TN/FP on category scores
    def maximize_f_score(self, y_expected, cat_scores, category, beta):
        # TODO: debug
        convergence_steps = 0

        def inv_f_score_func_wrapper(x):
            # global convergence_steps
            # convergence_steps += 1
            return -self.f_score_for_category(y_expected, cat_scores, category, x, beta)

        logging.info("Tuning f_score of cat %s for beta %s by %s scores" % (category, beta, len(cat_scores)))

        opt_threshold = minimize_scalar(fun=inv_f_score_func_wrapper, method="bounded", bounds=(0, 1)).x
        opt_score = -inv_f_score_func_wrapper(opt_threshold)
        logging.info("Threshold for category %s converged in %s steps to %s with f(beta)= %s"
                     % (category, -1, opt_threshold, opt_score))

        return opt_threshold

    # D:1)
    # series of probs, scalar of original_cat_threshold
    @staticmethod
    def transform_probs(cat_probs, original_cat_threshold):
        # relative ratios of spaces divided by a split of <0, 1> prob space by a value of doc prob
        # where space is determined by a distance of doc_i prob from original_cat_threshold
        target_threshold = general_search_threshold

        # the figure projects points into <0, 1> only with symmetric target_threshold = general_search_threshold
        probs_below_trhd = cat_probs[cat_probs < original_cat_threshold]
        probs_above_trhd = cat_probs[cat_probs >= original_cat_threshold]

        probs_below_trhd_ratio = (original_cat_threshold - probs_below_trhd) / original_cat_threshold
        probs_above_trhd_ratio = (probs_above_trhd - original_cat_threshold) / (1 - original_cat_threshold)

        probs_below_trhd_new = target_threshold - (probs_below_trhd_ratio * target_threshold)
        probs_above_trhd_new = target_threshold + (probs_above_trhd_ratio * (1 - target_threshold))

        return probs_below_trhd_new.append(probs_above_trhd_new)

    # D:2)
    # normalizes the category scores so that the relevance of documents is easily evaluated using the unified threshold
    # to distinguish non-relevant from relevant
    # sensitivity (=precision/recall ratio) of a search can be customized by parametrized f-score for category
    def tune_cat_scores(self, cat_scores, category):
        # moves the scores so that the threshold gets to general_search_threshd, and all scores are scaled in <0, 1>
        norm_cat_scores = self.transform_probs(cat_scores, self.cats_original_thresholds[category])
        return norm_cat_scores

    # E:0) General linear scaling function provider
    def get_scaling_func(self, input_intvl, target_intvl, lower_border=0.1):
        targets = np.array(target_intvl)
        coef_matrix = np.matrix([[input_intvl[0], 1],
                                 [input_intvl[1], 1]])
        ab_linear_coefs = np.linalg.solve(coef_matrix, targets)

        return lambda x: ab_linear_coefs[0] * x + ab_linear_coefs[1] \
            if ab_linear_coefs[0] * x + ab_linear_coefs[1] > 0 else lower_border

    # F)
    # computes beta that is empirically presumed to convey the expectations of the search engine for given categories
    # we want to weight beta as log-dependent to categories sizes
    # this way we'll prefer a weight of precision for bigger categories and recall for categories with little content
    def beta_for_categories_provider(self, y_expected):
        # function retrieves beta in interval <0.2, 5>
        # params were set to log-approximate the interval for categories between sizes of <100, 20 000> content

        cats_sizes = y_expected.value_counts()
        logging.info("Categories size: \n%s" % cats_sizes)

        # normalization function - linear function mapping interval <1000, 20000> (= category size) to <5, 0.2>:
        cats_order_by_size = pd.Series(data=cats_sizes.values.argsort(), index=cats_sizes.index)

        # TODO: betas for categories are not as nice as we might like - that might need some non-linear function
        scaling_f = self.get_scaling_func(input_intvl=[cats_order_by_size.min(), cats_order_by_size.max()],
                                          target_intvl=[5, 0.2])

        # lower betas weight more significantly precision, higher weight more recall
        cats_betas = cats_order_by_size.apply(scaling_f)
        logging.warn("Categories f-score betas as scaled by cat sizes: \n%s" % cats_betas)

        return cats_betas

    # D:3)
    # gets df of scores of some content for all categories and expected labels
    # will tune the scores of categories so that the same threshold for all categories can be applied
    # and still the selected content will qualitatively persist the maximized F-score
    def tune_all_scores(self, scores_df):
        logging.info("Tuning probs of %s docs" % (len(scores_df)))

        norm_scores_df = pd.DataFrame()
        for cat_label in scores_df.keys().unique():
            norm_scores_df[cat_label] = self.tune_cat_scores(scores_df[cat_label], cat_label)
        return norm_scores_df

    # E:0) Infers linearly-scaled weights for each category by its size
    @staticmethod
    def weighted_combine_cats_predictions(expected_labels, cats_performance):
        def scaling_f(cat_size):
            return 1 / np.math.sqrt(cat_size)

        # categories must be weighted decreasingly by size
        performance_scalars = expected_labels.value_counts().apply(scaling_f) * expected_labels.value_counts()
        logging.warn("Cats performance weights: \n%s" % performance_scalars)

        combined_performance = np.average(cats_performance, weights=performance_scalars)
        return combined_performance

    # E:1) computes a performance from the scope of a search results
    # considering as relevant the documents above the fixed threshold (general_search_threshold)
    # NOTE: both y_expected and scores_df must have same length and matching index
    def evaluate(self, y_expected, scores_df):
        score_df_normalized = self.tune_all_scores(scores_df)
        cat_fscore_betas = self.beta_for_categories_provider(y_expected)
        cat_labels = pd.Series(y_expected.unique())
        cats_performance = cat_labels.apply(lambda cat_label: self.f_score_for_category(y_expected,
                                                                                        score_df_normalized[cat_label],
                                                                                        cat_label,
                                                                                        general_search_threshold,
                                                                                        cat_fscore_betas[cat_label]))
        cats_performance.index = cat_labels.values
        combined_cat_performance = self.weighted_combine_cats_predictions(y_expected, cats_performance)

        return combined_cat_performance

    # Test method for standalone functionality evaluation
    @staticmethod
    def debug_load_pickled_scores(scores_df_path, y_expected_path):
        logging.info("Loading scores and y_expected from (%s, %s)" % (scores_df_path, y_expected_path))
        with open(scores_df_path, "r") as pickle_file:
            scores_df = cPickle.load(pickle_file)
            scores_df.index = np.array(range(len(scores_df)))

        with open(y_expected_path, "r") as pickle_file:
            all_y_expected = cPickle.load(pickle_file)
            all_y_expected.index = np.array(range(len(scores_df)))

        return scores_df, all_y_expected


# TODO: for TEST:
# scores_df, y_expected = load_pickled_scores("temp_pickled_scores_df.dump", "temp_pickled_y_expected.dump")
# performance = evaluate(y_expected, scores_df)
# logging.warn("Overall performance: %s" % performance)
