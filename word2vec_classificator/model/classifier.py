# inspired by
# https://github.com/TaddyLab/gensim/blob/deepir/docs/notebooks/deepir.ipynb
# https://arxiv.org/pdf/1504.07295.pdf
# https://arxiv.org/pdf/1405.4053v2.pdf

from gensim.models import Word2Vec
import multiprocessing
import pandas as pd
import numpy as np
from copy import deepcopy

import common.parsing_utils as parsing


class W2VClassifier:
    content_categories = ["portal", "amq", "webserver", "fsw", "eap"]
    # content_categories = ["portal", "amq", "webserver", "fsw"]
    content_basepath = None
    basepath_suffix = "_content.csv"
    all_content_df = None

    generic_vocab_model = None
    category_models = dict()

    def __init__(self, content_basepath="../../data/content", basepath_suffix="_content.csv", content_categories=None):
        self.content_basepath = content_basepath
        self.basepath_suffix = basepath_suffix
        if content_categories:
            self.content_categories = content_categories

    def init_vocab_model(self):
        self.generic_vocab_model = Word2Vec(
            workers=multiprocessing.cpu_count(),
            iter=30,  # iter = sweeps of SGD through the data; more is better
            hs=1, negative=0  # we only have scoring for the hierarchical softmax setup
        )

        self.all_content_df = self.get_content_as_dataframe()
        # training attributes sorted by relevance
        all_content_sens = parsing.select_training_content(self.all_content_df)

        self.generic_vocab_model.build_vocab(all_content_sens)

    def init_model_from_dataframe(self, cat_label, cat_sentences_df=None):
        # expects dataframe of only content selected for training
        if not self.generic_vocab_model:
            self.init_vocab_model()

        if cat_sentences_df is None:
            cat_content_df = self.get_content_as_dataframe(cat_label=cat_label)
            cat_sentences_df = parsing.select_training_content(cat_content_df)

        self.category_models[cat_label] = deepcopy(self.generic_vocab_model)
        # cat_content_sens = parsing.select_training_content(cat_sentences_df)
        # TODO: check if we really train on sentences, not words
        self.category_models[cat_label].train(cat_sentences_df.values, total_examples=len(cat_sentences_df))

    def init_all_models(self, content_df=None):
        if not content_df:
            content_df = self.get_content_as_dataframe()
        for cat_label in self.content_categories:
            self.category_models[cat_label] = deepcopy(self.generic_vocab_model)
            # cat_content_df = self.get_content_as_dataframe(cat_label=cat_label)
            cat_content_sens = parsing.select_training_content(content_df)

            self.category_models[cat_label].train(cat_content_sens, total_examples=len(cat_content_sens))

    def get_content_as_dataframe(self, cat_label=None):
        if not cat_label:
            # retrieve all content of all known categories
            all_content = pd.DataFrame(
                columns="sys_title,sys_description,source,sys_content_plaintext,target".split(","))
            for cat_label in self.content_categories:
                new_content = pd.read_csv("%s/%s%s" % (self.content_basepath, cat_label, self.basepath_suffix),
                                          na_filter=False)
                all_content = all_content.append(new_content)
            return all_content
        else:
            # retrieve only one cat_label content
            cat_content = pd.read_csv("%s/%s%s" % (self.content_basepath, cat_label, self.basepath_suffix),
                                      na_filter=False)
            return cat_content

    # score the semantic similarity of the document against the models of the trained categories
    # document param is supposed to be an instance of Document class
    def score_content_for_categories(self, content_sentences):
        # content_sentences = parsing.content_to_sentence_split(content_plaintext)
        try:
            scores_vector = [model.score(content_sentences, total_sentences=len(content_sentences)) for model in self.category_models.values()]
        except UnboundLocalError:
            print("UnboundLocalError on %s" % content_sentences)
            return {label: 0 for label in self.content_categories}
        # TODO: normalization sucks
        # normalize with sum of the vector to better interpretability - equivalent in terms of ordering
        scores_vector = map(lambda score: score/sum(scores_vector), scores_vector)
        label_scores = {label: np.mean(scores_vector[self.content_categories.index(label)]) for label in self.content_categories}
        return label_scores

    # returns the single category label with the highest score
    def best_match(self, docs_sentence):
        docs_scores_matrix = self.score_documents_for_categories(docs_sentence)
        best_matching_indices = docs_scores_matrix.apply(lambda scores: np.argmax(scores), axis=1)
        best_matching_labels = best_matching_indices.apply(lambda cat_index: self.content_categories[cat_index])
        return best_matching_labels

    # classify a vector of contents
    # TODO: need to be done also predict for documents - averaging the results on its sentences - check!
    def predict_all(self, docs_sentence):
        docs_sentence["y"] = self.best_match(docs_sentence)
        return docs_sentence

    # TODO: remove
    def predict_sentence(self, sentences_df):
        sentences_df["y"] = sentences_df.apply(lambda content: self.best_match(content))
        return sentences_df

    def score_documents_for_categories(self, docs):
        # score() takes a list [s] of sentences here; could also be a sentence generator
        sentlist = [s for d in docs for s in d]
        # the log likelihood of each sentence in this review under each w2v representation
        llhd = np.array([m.score(sentlist, len(sentlist)) for m in self.category_models.values()])
        # now exponentiate to get likelihoods,
        lhd = np.exp(llhd - llhd.max(axis=0))  # subtract row max to avoid numeric overload
        # normalize across models (stars) to get sentence-star probabilities
        prob = pd.DataFrame((lhd / lhd.sum(axis=0)).transpose())
        # and finally average the sentence probabilities to get the review probability
        prob["doc"] = [i for i, d in enumerate(docs) for s in d]
        prob = prob.groupby("doc").mean()
        return prob

