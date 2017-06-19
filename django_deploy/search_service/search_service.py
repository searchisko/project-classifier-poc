# TODO: resolve $PYTHONPATH - currently set, might not be the nicest way of resolving dependencies
# export PYTHONPATH=/home/michal/Documents/Projects/ml/project-classifier-poc/project-classifier-poc/django_deploy/search_service:/home/michal/Documents/Projects/ml/project-classifier-poc/project-classifier-poc/django_deploy/search_service/dependencies

import os

import logging
import multiprocessing
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

import pandas as pd
import numpy as np

from dependencies.doc2vec_wrapper import D2VWrapper
from dependencies.scores_tuner import ScoreTuner
from dependencies import parsing_utils as parsing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class RelevanceSearchService:
    model_categories = None
    d2v_wrapper = None
    vector_classifier = None
    score_tuner = None
    trained = False
    classifier_name = "logistic_regression.mod"
    default_model_dir = "/home/michal/Documents/Projects/ml/project-classifier-poc/project-classifier-poc/django_deploy/search_service/persisted_model_prod"
    service_meta = dict()

    def __init__(self):

        self.service_meta["init_timestamp"] = datetime.datetime.utcnow()

        self.service_meta["model_reload_dir"] = None
        self.service_meta["model_reload_timestamp"] = None
        self.service_meta["model_persist_dir"] = None
        self.service_meta["model_persist_timestamp"] = None

        self.service_meta["model_train_start_timestamp"] = None
        self.service_meta["model_train_end_timestamp"] = None
        self.service_meta["model_train_src_dir"] = None

        self.service_meta["model_eval_result"] = None
        self.service_meta["score_requests_counter"] = 0

        self.d2v_wrapper = D2VWrapper()

    @staticmethod
    def _get_classifier_instance():
        return LogisticRegression(C=0.22, solver="sag", multi_class='ovr',
                                  n_jobs=multiprocessing.cpu_count(), max_iter=1000)

    """
    Scores as inferred by a classifier are later used to tune the categories probs
    so that the search can reach optimal performance by using the fixed threshold.
    Content scoring is done in 10-fold CV, inferring the scores of 1/10 of content
    using the predict_proba of a classifier trained on other 9/10 of content
    """

    def _score_train_content(self, doc_vectors, y):
        docs_scores = pd.DataFrame(columns=self.model_categories)
        # TODO: splits
        splits = 5

        strat_kfold = StratifiedKFold(n_splits=splits, shuffle=True)
        logging.info("Gathering training content scores as infered by a classifier %s in %s splits"
                     % (str(self._get_classifier_instance().__class__()), splits))

        for train_doc_indices, test_doc_indices in strat_kfold.split(doc_vectors, y):
            split_vector_classifier = self._get_classifier_instance()
            logging.info("Fitting split classifier")
            split_vector_classifier.fit(doc_vectors.iloc[train_doc_indices], y.iloc[train_doc_indices])
            logging.info("Predicting split probs")
            inferred_scores = split_vector_classifier.predict_proba(doc_vectors.iloc[test_doc_indices])
            categories_ordered = list(split_vector_classifier.classes_)
            inferred_scores_df = pd.DataFrame(data=inferred_scores, columns=categories_ordered, index=test_doc_indices)

            docs_scores = docs_scores.append(inferred_scores_df)

        return docs_scores.sort_index()

    """
    Optimize the categories thresholds of score_tuner to maximize the combined categories' f-score
    to be used for score tuning on a new content requested to be scored.
    """
    def _train_score_tuner(self, doc_vectors, y, score_tuner):
        scores_df = self._score_train_content(doc_vectors, y)
        score_tuner.train_categories_thresholds(y, scores_df)
        logging.info("Score tuner trained")
        return score_tuner

    """
    Train all required system models on a content in a given train_content_dir folder.
    The folder should contain categories' content in csv-formed files <category>_content.csv.
    The files should hold the structure as produced by downloader_automata.py
    """
    def train(self, train_content_dir):
        # init d2v_wrapper with categorized documents
        self.service_meta["model_train_start_timestamp"] = datetime.datetime.utcnow()

        self.d2v_wrapper.init_model_vocab(content_basepath=train_content_dir, drop_short_docs=10)
        # categories are inferred by target directory containment in vocab init of d2v
        self.model_categories = self.d2v_wrapper.content_categories
        # train d2v model, infer docs vectors and train adjacent classifier
        # TODO epochs
        self.d2v_wrapper.train_model(epochs=3)

        doc_vectors_labeled = self.d2v_wrapper.infer_vocab_content_vectors()
        doc_vectors = doc_vectors_labeled.iloc[:, :-1]
        y = doc_vectors_labeled.iloc[:, -1]

        # superior classifier training
        logging.info("Fitting classifier on %s docs vectors" % len(doc_vectors_labeled))
        self.vector_classifier = self._get_classifier_instance()
        self.vector_classifier.fit(doc_vectors, y)

        logging.info("Model %s fitted" % self.vector_classifier.__class__)

        # scores tuning
        self.score_tuner = self._train_score_tuner(doc_vectors, y, ScoreTuner())

        self.service_meta["model_train_end_timestamp"] = datetime.datetime.utcnow()
        logging.info("Model training finished")
        self.service_meta["model_train_src_dir"] = train_content_dir

    def persist_trained_model(self, persist_dir=default_model_dir):
        if self.service_meta["model_train_src_dir"] is None:
            logging.error("Service models have not been trained yet. Run self.train(train_content_path) first")

        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        # use d2v_wrapper persistence and classifier persistence
        self.d2v_wrapper.persist_trained_wrapper(persist_dir, model_only=True)
        # persist all dependent objects to a single, newly created directory (persist_dir)

        classifier_path = persist_dir + "/" + self.classifier_name
        logging.info("Pickling classifier to %s" % classifier_path)
        joblib.dump(self.vector_classifier, classifier_path)

        tuner_path = persist_dir + "/score_tuner.mod"
        logging.info("Pickling score tuner to %s" % tuner_path)
        joblib.dump(self.score_tuner, tuner_path)

        self.service_meta["model_persist_dir"] = persist_dir
        self.service_meta["model_persist_timestamp"] = datetime.datetime.utcnow()

        meta_path = persist_dir + "/metadata.mod"
        logging.info("Pickling model metadata to %s" % meta_path)
        joblib.dump(self.service_meta, meta_path)

    """load previously trained model from the persist_dir"""

    def load_trained_model(self, persist_dir=default_model_dir):
        if self.service_meta["model_train_src_dir"] is not None:
            logging.warn("Overriding the loaded model from %s" % self.service_meta["model_train_src_dir"])

        self.d2v_wrapper.load_persisted_wrapper(persist_dir, model_only=True)
        self.model_categories = self.d2v_wrapper.content_categories

        classifier_path = persist_dir + "/" + self.classifier_name
        logging.info("Loading pickled classifier from %s" % classifier_path)
        self.vector_classifier = joblib.load(classifier_path)

        tuner_path = persist_dir + "/score_tuner.mod"
        logging.info("Loading score tuner from %s" % tuner_path)
        self.score_tuner = joblib.load(tuner_path)

        meta_path = persist_dir + "/metadata.mod"
        logging.info("Loading model metadata from %s" % meta_path)
        self.service_meta = joblib.load(meta_path)

        self.service_meta["model_reload_dir"] = persist_dir
        self.service_meta["model_reload_timestamp"] = datetime.datetime.utcnow()

    """
    Scores previously unseen single doc towards categories of train_content
    """

    def score_doc(self, doc_id, doc_header, doc_content):
        return self.score_docs_bulk([doc_id], [doc_header], [doc_content])

    """
    Scores an iterable of docs of headers, content and unique ids in bulk.
    Faster than calling score_doc for each document
    """

    def score_docs_bulk(self, doc_ids, doc_headers, doc_contents):
        logging.info("Requested docs: %s for scoring" % list(doc_ids))
        # check integrity
        if type(doc_ids) is not pd.Series:
            doc_ids = pd.Series(doc_ids)

        if doc_ids.unique().__len__() < len(doc_ids):
            raise ValueError("doc_ids parameter must contain unique values to further specifically identify docs.")

        if not all([self.d2v_wrapper, self.vector_classifier, self.score_tuner]):
            logging.warning("First service call. Will try to load model from self.default_model_dir")
            # try:
            self.load_trained_model()
            # except IOError:
            #     raise UserWarning(
            #             "Depended models not found in self.default_model_dir = %s."
            #             "Please train and export the model to the given directory "
            #             "so it can be loaded at first score request" % self.default_model_dir)

        # preprocess content
        logging.info("Docs %s: preprocessing" % list(doc_ids))
        doc_objects_series = pd.Series(data=parsing.tagged_docs_from_plaintext(doc_contents, doc_headers,
                                                                       pd.Series([None] * len(doc_ids))))
        # vectorize content
        logging.info("Docs %s: vectorizing using trained Doc2Vec model" % list(doc_ids))
        new_docs_vectors = self.d2v_wrapper.infer_content_vectors(docs=doc_objects_series)

        # classify = predict probabilities for known categories
        logging.info("Docs %s: classifying using %s classifier" % (list(doc_ids), self.classifier_name))
        new_content_probs = self.vector_classifier.predict_proba(X=new_docs_vectors)
        cats_ordered = list(self.vector_classifier.classes_)
        new_content_probs_df = pd.DataFrame(data=new_content_probs, columns=cats_ordered, index=doc_ids)

        # tune categories probabilities to scores
        logging.info("Docs %s: tuning probs according to optimized cats thresholds:\n%s"
                     % (list(doc_ids), self.score_tuner.cats_original_thresholds))
        new_content_scores_tuned = self.score_tuner.tune_new_docs_scores(new_content_probs_df)

        self.service_meta["score_requests_counter"] += len(doc_ids)
        return new_content_scores_tuned, cats_ordered

    """
    Evaluates performance of the deployed classifier in CV manner.
    The results are memorized in self.service_meta["model_eval_result"] after the test finishes
    The test might take tens of minutes depending on a number of folds.
    """

    def evaluate_performance(self, folds=5, target_search_threshold=0.5):
        logging.info("Performance evaluation on service \n%s" % self.service_meta["init_timestamp"])
        vocab_docs = self.d2v_wrapper.infer_vocab_content_vectors()
        vocab_docs_vectors = vocab_docs.iloc[:, :-1]
        vocab_y = vocab_docs.iloc[:, -1]

        performance = []
        cats_performance = pd.DataFrame(columns=self.model_categories)

        strat_kfold = StratifiedKFold(n_splits=folds, shuffle=True)

        for train_doc_indices, test_doc_indices in strat_kfold.split(vocab_docs_vectors, vocab_y):
            self.vector_classifier.fit(vocab_docs_vectors.iloc[train_doc_indices], vocab_y.iloc[train_doc_indices])

            inferred_scores = self.vector_classifier.predict_proba(vocab_docs_vectors.iloc[test_doc_indices])
            categories_ordered = list(self.vector_classifier.classes_)
            inferred_scores_df = pd.DataFrame(data=inferred_scores, columns=categories_ordered, index=test_doc_indices)
            split_performance = self.score_tuner.evaluate(vocab_y.iloc[test_doc_indices], inferred_scores_df)

            logging.info("Performance of this split in system performance evaluation: %s" % split_performance)
            performance.append(split_performance)

            tuned_scores_df = self.score_tuner.tune_all_scores(inferred_scores_df)

            # categories performance analysis
            categories_fscore_betas = self.score_tuner.beta_for_categories_provider(vocab_y)
            split_cats_performance = pd.Series(self.model_categories, index=cats_performance.columns).apply(
                lambda cat_label: self.score_tuner.f_score_for_category(vocab_y,
                                                                        tuned_scores_df[cat_label],
                                                                        cat_label,
                                                                        target_search_threshold,
                                                                        categories_fscore_betas[cat_label]))

            logging.info("Categories performance of this split on threshold %s: \n%s" % (target_search_threshold,
                                                                                         split_cats_performance))
            cats_performance.loc[len(cats_performance)] = split_cats_performance

        logging.info("Overall performance results of search with separate threshold: %s:" % target_search_threshold)
        logging.info("Splits performance: \n%s" % performance)
        logging.info("Splits mean performance: \n%s" % np.mean(performance))
        logging.info("Categories splits performance: \n%s" % cats_performance)
        logging.info("Categories mean performance: \n%s" % cats_performance.apply(np.mean, axis=0))

        self.service_meta["model_eval_result"] = {"test_finish_time": datetime.datetime.utcnow(),
                                                  "mean_performance": np.mean(performance),
                                                  "categories_mean_performance": cats_performance.apply(np.mean, axis=0)}
