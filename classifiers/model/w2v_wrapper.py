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


class W2VWrapper:
    # content_categories might be set from outer scope
    content_categories = None
    content_basepath = None
    basepath_suffix = "_content.csv"
    all_content = None
    all_content_target = None

    generic_vocab_model = None
    category_models = dict()

    def __init__(self, content_basepath, basepath_suffix="_content.csv", content_categories=None, vector_length=300, training_algo="cbow"):
        self.content_basepath = content_basepath
        self.basepath_suffix = basepath_suffix
        if content_categories:
            self.content_categories = content_categories

        self.generic_vocab_model = Word2Vec(
            workers=multiprocessing.cpu_count(),
            iter=30,  # iter = sweeps of SGD through the data; more is better
            hs=1, negative=0,  # we only have scoring for the hierarchical softmax setup
            size=vector_length,
            sg=0 if training_algo == "cbow" else 1
        )

    def init_vocab_model(self, given_content_series=None, given_content_targets=None, drop_short_docs=None):
        if given_content_series is None and given_content_targets is None:
            all_content_df, self.all_content_target = self.get_content_as_dataframe()
            # training attributes sorted by relevance
            self.all_content = parsing.select_training_content(all_content_df, sent_split=False)
            if drop_short_docs is not None:
                short_docs_filter = self.all_content.apply(lambda doc: len(doc) >= drop_short_docs)
                self.all_content = self.all_content[short_docs_filter]
                self.all_content_target = self.all_content_target[short_docs_filter]

                self.all_content.index = np.arange(len(self.all_content))
                self.all_content_target.index = np.arange(len(self.all_content))
        else:
            self.all_content = given_content_series
            self.all_content_target = given_content_targets

        self.generic_vocab_model.build_vocab(self.all_content)

    def init_model_from_dataframe(self, cat_label, cat_sentences_df=None):
        # expects dataframe of only content selected for training
        if not self.generic_vocab_model:
            self.init_vocab_model()

        if cat_sentences_df is None:
            cat_content_df = self.get_content_as_dataframe(cat_label=cat_label)
            cat_sentences_df = parsing.select_training_content(cat_content_df)

        self.category_models[cat_label] = deepcopy(self.generic_vocab_model)
        # cat_content_sens = parsing.select_training_content(cat_sentences_df)
        self.category_models[cat_label].train(cat_sentences_df.values, total_examples=len(cat_sentences_df))

    def train_categories_models(self):
        for cat_label in self.content_categories:
            self.category_models[cat_label] = deepcopy(self.generic_vocab_model)

            cat_content_series = self.all_content[self.all_content_target == cat_label]
            self.category_models[cat_label].train(cat_content_series, total_examples=len(cat_content_series))

    def get_content_as_dataframe(self, cat_label=None):
        if not cat_label:
            # retrieve all content of all known categories
            all_content_mapping = []
            all_content = pd.DataFrame(
                columns="sys_title,sys_description,source,sys_content_plaintext,target".split(","))

            for cat_label in self.content_categories:
                new_content = pd.read_csv("%s/%s%s" % (self.content_basepath, cat_label, self.basepath_suffix),
                                          na_filter=False, error_bad_lines=False)

                new_content = new_content.drop_duplicates()
                all_content = all_content.append(new_content)
                all_content_mapping.extend([cat_label]*len(new_content))

            return all_content, pd.Series(all_content_mapping)
        else:
            # retrieve only one cat_label content
            cat_content = pd.read_csv("%s/%s%s" % (self.content_basepath, cat_label, self.basepath_suffix),
                                      na_filter=False)
            return cat_content, pd.Series([cat_label]*len(cat_content))

    # returns the single category label with the highest score
    def best_match(self, docs_sentence):
        docs_scores_matrix = self.score_documents_for_categories(docs_sentence)
        best_matching_indices = docs_scores_matrix.apply(lambda scores: np.argmax(scores), axis=1)
        best_matching_labels = best_matching_indices.apply(lambda cat_index: self.category_models.keys()[cat_index])
        return best_matching_labels

    # classify a vector of contents
    def predict_all(self, docs_sentence):
        docs_sentence["y"] = self.best_match(docs_sentence.apply(lambda doc: [doc]).values)
        return docs_sentence

    def score_documents_for_categories(self, docs):
        # score() takes a list [s] of sentences here; could also be a sentence generator
        sentlist = [s for d in docs for s in d]
        # the log likelihood of each sentence in this review under each w2v representation
        llhd = np.array([m.score(sentlist, len(sentlist)) for m in self.category_models.values()])
        # now exponentiate to get likelihoods,
        lhd = np.exp(llhd - llhd.max(axis=0))  # subtract row max to avoid numeric overload
        # normalize across models (stars) to get sentence-star probabilities
        prob = pd.DataFrame((lhd).transpose())
        # and finally average the sentence probabilities to get the review probability
        prob["doc"] = [i for i, d in enumerate(docs) for s in d]
        prob = prob.groupby("doc").mean()
        return prob

    def get_doc_content(self, doc_index, word_split=True):
        if word_split:
            return self.all_content[doc_index]
        else:
            return parsing.content_from_words(word_list=self.all_content[doc_index])

