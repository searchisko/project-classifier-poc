import os
from os import listdir
from os.path import isfile, join

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
    service_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    service_image_dir = service_dir + "/" + "trained_service_prod"
    service_meta = dict()
    minimized_persistence = True

    # debug global vars
    # might be independently pickled after training to further examine
    train_scores_df = None

    def __init__(self, image_dir=None):
        if image_dir is not None:
            self.service_image_dir = self.service_dir + "/" + image_dir

        self.service_meta["init_timestamp"] = datetime.datetime.utcnow()

        self.service_meta["model_reload_dir"] = None
        self.service_meta["model_reload_timestamp"] = None
        self.service_meta["model_persist_dir"] = None
        self.service_meta["model_persist_timestamp"] = None

        self.service_meta["model_train_start_timestamp"] = None
        self.service_meta["model_train_end_timestamp"] = None
        self.service_meta["model_train_src"] = None

        self.service_meta["model_eval_result"] = None
        self.service_meta["score_requests_counter"] = 0

        self.d2v_wrapper = D2VWrapper(vector_length=800)

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
        # TODO: set splits
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
            inferred_scores_df = pd.DataFrame(data=inferred_scores, columns=categories_ordered, index=doc_vectors.iloc[test_doc_indices].index)

            docs_scores = docs_scores.append(inferred_scores_df)

        return docs_scores.sort_index()

    """
    Optimize the categories thresholds of score_tuner to maximize the combined categories' f-score
    to be used for score tuning on a new content requested to be scored.
    """
    def _train_score_tuner(self, doc_vectors, y, score_tuner):
        self.train_scores_df = self._score_train_content(doc_vectors, y)
        self.train_scores_df["y"] = y
        score_tuner.train_categories_thresholds(y, self.train_scores_df)
        logging.info("Score tuner trained")
        return score_tuner

    def _train_vector_classifier(self, X, y, classifier=None):
        if classifier is None:
            classifier = self._get_classifier_instance()
        # superior classifier training
        logging.info("Fitting classifier on %s docs vectors" % len(X))
        classifier.fit(X, y)

        logging.info("Model %s fitted" % classifier.__class__)
        return classifier

    def _train_d2v_wrapper(self):
        # categories are inferred by target directory containment in vocab init of d2v
        self.model_categories = self.d2v_wrapper.content_categories
        # train d2v model, infer docs vectors and train adjacent classifier
        # TODO set epochs
        self.d2v_wrapper.train_model(epochs=10)

        doc_vectors_labeled = self.d2v_wrapper.infer_vocab_content_vectors()
        doc_vectors = doc_vectors_labeled.iloc[:, :-1]
        y = doc_vectors_labeled.iloc[:, -1]

        return doc_vectors, y

    """TODO"""
    def train_on_docs(self, doc_ids, doc_headers, doc_contents, y):
        doc_contents = pd.Series(doc_contents)
        doc_headers = pd.Series(doc_headers)
        y = pd.Series(y)

        logging.info("Training service on docs: %s" % np.array(doc_ids))

        training_docs_objects = parsing.tagged_docs_from_plaintext(doc_contents, doc_headers, y)
        self.d2v_wrapper.init_vocab_from_docs(training_docs_objects)

        doc_vectors, y = self._train_d2v_wrapper()
        doc_vectors.index = y.index

        self.vector_classifier = self._train_vector_classifier(doc_vectors, y)
        self.score_tuner = self._train_score_tuner(doc_vectors, y, ScoreTuner())

        self.service_meta["model_train_end_timestamp"] = datetime.datetime.utcnow()
        logging.info("Model training finished")
        self.service_meta["model_train_src"] = "%s docs, %s categories" % (len(doc_ids), len(np.unique(y)))

    """
    Train all required system models on a content in a given train_content_dir folder.
    The folder should contain categories' content in csv-formed files <category>_content.csv.
    The files should hold the structure as produced by products_downloader.py
    """
    def train(self, train_content_dir):
        # init d2v_wrapper with categorized documents
        self.service_meta["model_train_start_timestamp"] = datetime.datetime.utcnow()

        self.d2v_wrapper.init_model_vocab(content_basepath=train_content_dir,
                                          drop_short_docs=10)

        doc_vectors, y = self._train_d2v_wrapper()

        self.vector_classifier = self._train_vector_classifier(doc_vectors, y)

        # scores tuning
        self.score_tuner = self._train_score_tuner(doc_vectors, y, ScoreTuner())

        self.service_meta["model_train_end_timestamp"] = datetime.datetime.utcnow()
        logging.info("Model training finished")
        self.service_meta["model_train_src"] = train_content_dir

    def persist_trained_model(self, persist_dir=None):
        if persist_dir is None:
            persist_dir = self.service_image_dir

        if self.service_meta["model_train_src"] is None:
            logging.error("Service models have not been trained yet. Run self.train(train_content_path) first")

        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        # use d2v_wrapper persistence and classifier persistence
        self.d2v_wrapper.persist_trained_wrapper(persist_dir, model_only=self.minimized_persistence)
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

    def load_trained_model(self, persist_dir=None):
        if persist_dir is None:
            persist_dir = self.service_image_dir

        if self.service_meta["model_train_src"] is not None:
            logging.warn("Overriding the loaded model from %s" % self.service_meta["model_train_src"])

        self.d2v_wrapper.load_persisted_wrapper(persist_dir, model_only=self.minimized_persistence)

        classifier_path = persist_dir + "/" + self.classifier_name
        logging.info("Loading pickled classifier from %s" % classifier_path)
        self.vector_classifier = joblib.load(classifier_path)

        self.model_categories = self.vector_classifier.classes_

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

    def score_doc(self, doc_id, doc_header, doc_content, preprocess_method=parsing.preprocess_text):
        return self.score_docs_bulk([doc_id], [doc_header], [doc_content], preprocess_method=preprocess_method)

    """
    Scores an iterable of docs of headers, content and unique ids in bulk.
    Faster than calling score_doc for each document
    """

    def score_docs_bulk(self, doc_ids, doc_headers, doc_contents, preprocess_method=parsing.preprocess_text):
        logging.info("Requested docs: %s for scoring" % np.array(doc_ids))

        doc_ids = pd.Series(doc_ids)
        # check integrity
        if doc_ids.unique().__len__() < len(doc_ids):
            raise ValueError("doc_ids parameter must contain unique values to further specifically identify docs.")

        if not all([self.d2v_wrapper, self.vector_classifier, self.score_tuner]):
            logging.warning("First service call. Will try to load model from self.service_image_dir")
            try:
                self.load_trained_model()
            except IOError:
                raise UserWarning(
                        "Depended models not found in self.service_image_dir = %s."
                        "Please train and export the model to the given directory "
                        "so it can be loaded at first score request" % self.service_image_dir)

        # preprocess content
        logging.info("Docs %s: preprocessing" % np.array(doc_ids))
        doc_objects_series = pd.Series(data=parsing.tagged_docs_from_plaintext(doc_contents, doc_headers,
                                                                               pd.Series([None] * len(doc_ids)),
                                                                               preprocess_method=preprocess_method))
        # vectorize content
        logging.info("Docs %s: vectorizing using trained Doc2Vec model" % np.array(doc_ids))
        new_docs_vectors = self.d2v_wrapper.infer_content_vectors(docs=doc_objects_series)

        # classify = predict probabilities for known categories
        logging.info("Docs %s: classifying using %s classifier" % (np.array(doc_ids), self.classifier_name))
        new_content_probs = self.vector_classifier.predict_proba(X=new_docs_vectors)
        cats_ordered = list(self.vector_classifier.classes_)
        new_content_probs_df = pd.DataFrame(data=new_content_probs, columns=cats_ordered, index=doc_ids)

        # tune categories probabilities to scores
        logging.info("Docs %s: tuning probs according to optimized cats thresholds:\n%s"
                     % (np.array(doc_ids), self.score_tuner.cats_original_thresholds))
        new_content_scores_tuned = self.score_tuner.tune_new_docs_scores(new_content_probs_df)

        self.service_meta["score_requests_counter"] += len(doc_ids)

        return new_content_scores_tuned

    """
    Deep-evaluates the performance of the classifier with pre-defined service configuration.
    To assure the distinct train-test sets, the evaluation does not use the instantiated service, but rather train the
    new one and test it on left out data set in CV manner.
    Eval data set is attempted to be found in a given eval_content_dir folder.
    The results are memorized in self.service_meta["model_eval_result"] after the test finishes
    The test might take tens of minutes depending on a number of folds and a size of eval data set.
    """

    def evaluate_performance(self, eval_content_dir="../../data/content/prod_sample", folds=5, target_search_threshold=0.5):
        logging.info("EVAL: evaluation routine started")
        logging.info("EVAL: collecting evaluation content from %s" % eval_content_dir)

        basepath_suffix = "_content.csv"
        # infer the trained categories according to the files in directory
        dir_files = [f for f in listdir(eval_content_dir) if isfile(join(eval_content_dir, f))]
        content_categories = map(lambda dir_file_path: dir_file_path
                                 .replace(eval_content_dir, "")
                                 .replace(basepath_suffix, ""), dir_files)

        logging.info("EVAL: Found %s categories to train on: %s" % (len(content_categories), np.array(content_categories)))

        # parse the content into dataframe
        all_content_df = parsing.get_content_as_dataframe(eval_content_dir, basepath_suffix, content_categories).drop_duplicates()

        doc_content = all_content_df["sys_content_plaintext"]
        doc_headers = parsing.select_headers(all_content_df).apply(
            lambda content: "" if np.any(pd.isnull(content)) else content) \
            .apply(lambda word_list: parsing.content_from_words(word_list))

        doc_content.index = doc_headers.index
        y = all_content_df["target"]

        docs_df = pd.DataFrame(columns=["content", "headers"], index=doc_content.index)
        docs_df["content"] = doc_content
        docs_df["headers"] = doc_headers
        docs_df["y"] = y

        # drop duplicates from the set
        docs_df = docs_df[~docs_df.duplicated(subset=["headers", "content"])]

        # EVAL phase:
        # split the content into selected (train/dev)/eval pieces and gather the documents scores
        # inferred from the service trained on distinct content

        performance = []
        cats_performance = pd.DataFrame(columns=content_categories)
        # collecting statistics of categories performance based on the system scoring

        strat_kfold = StratifiedKFold(n_splits=folds, shuffle=True)
        logging.info("EVAL: Gathering training content scores in %s splits" % folds)

        test_docs_scores = pd.DataFrame(columns=content_categories + ["y"])

        for train_doc_indices, test_doc_indices in strat_kfold.split(docs_df, docs_df["y"]):
            logging.info("EVAL: Initializing new RelevanceSearchService")
            eval_service = RelevanceSearchService()

            train_docs_df = docs_df.iloc[train_doc_indices]
            test_docs_df = docs_df.iloc[test_doc_indices]

            logging.info("EVAL: training service")
            eval_service.train_on_docs(doc_ids=train_docs_df.index, doc_contents=train_docs_df["content"],
                                       doc_headers=train_docs_df["headers"], y=train_docs_df["y"])

            logging.info("EVAL: scoring")
            test_docs_scores = eval_service.score_docs_bulk(doc_ids=test_docs_df.index,
                                                            doc_contents=test_docs_df["content"],
                                                            doc_headers=test_docs_df["headers"])
            test_docs_scores["y"] = test_docs_df["y"]

            logging.info("EVAL: inferred %s scores" % len(test_docs_scores))
            test_docs_scores = test_docs_scores.append(test_docs_scores)

            logging.info("EVAL: gathering stats of split service performance")
            split_performance = eval_service.score_tuner.evaluate_trained(test_docs_df["y"], test_docs_scores)

            logging.warn("EVAL: Performance of this split in system performance evaluation: %s" % split_performance)
            performance.append(split_performance)

            # categories performance analysis
            categories_fscore_betas = eval_service.score_tuner.beta_for_categories_provider(train_docs_df["y"])
            split_cats_performance = pd.Series(content_categories, index=content_categories).apply(
                lambda cat_label: eval_service.score_tuner.f_score_for_category(test_docs_df["y"],
                                                                                test_docs_scores[cat_label],
                                                                                cat_label,
                                                                                target_search_threshold,
                                                                                categories_fscore_betas[cat_label]))

            logging.info("Categories performance of this split on threshold %s: \n%s" % (target_search_threshold,
                                                                                         split_cats_performance))
            cats_performance.loc[len(cats_performance)] = split_cats_performance

        # evaluate the inferred scores
        logging.info("Overall performance results of search with separate threshold: %s:" % target_search_threshold)
        logging.info("Splits performance: \n%s" % performance)
        logging.info("Splits mean performance: \n%s" % np.mean(performance))
        logging.info("Categories splits performance: \n%s" % cats_performance)
        logging.info("Categories mean performance: \n%s" % cats_performance.apply(np.mean, axis=0))

        self.service_meta["model_eval_result"] = {"test_finish_time": datetime.datetime.utcnow(),
                                                  "mean_performance": np.mean(performance),
                                                  "categories_mean_performance": cats_performance.apply(np.mean, axis=0)}
