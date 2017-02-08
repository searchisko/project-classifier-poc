# https://radimrehurek.com/gensim/models/word2vec.html

from gensim.models import Word2Vec
import multiprocessing
import pandas as pd
import numpy as np


class VocabularyModel:

    model = Word2Vec(
        workers=multiprocessing.cpu_count(),
        iter=30,  # iter = sweeps of SGD through the data; more is better
        hs=1, negative=0  # we only have scoring for the hierarchical softmax setup
    )

    params = dict()

    def __init__(self, params):
        self.params = params
        pass

    # replace the default model instance with a model with built vocabulary, or even trained on relevant content
    def set_built_model(self, built_model):
        self.model = built_model

    # aggregates content of the given documents into a list of its' sentences
    @staticmethod
    def aggregate_docs_content(documents):
        sen_list = pd.Series()
        for doc in documents:
            sen_list = sen_list.append(doc.sentence_list)
        return sen_list

    def build_vocab(self, sentences):
        # sentences = self.aggregate_docs_content(documents)
        self.model.build_vocab(sentences)

    def train_on_cat_sentences(self, sentences):
        # sentences = self.aggregate_docs_content(documents)
        self.model.train(sentences=sentences, total_examples=len(sentences))

    # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.score
    # normalization as in:
    # https://github.com/TaddyLab/gensim/blob/deepir/docs/notebooks/deepir.ipynb
    def score(self, sentence_list):
        log_sen_probs = np.array(self.model.score(sentence_list))
        norm_sen_probs = np.exp(log_sen_probs-log_sen_probs.max(axis=0))
        # TODO: normalize against sum of other categories scores
        return norm_sen_probs.mean()
