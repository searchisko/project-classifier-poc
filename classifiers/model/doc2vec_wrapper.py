# TODO:

# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
# https://groups.google.com/forum/#!msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ
import random

from gensim.models import doc2vec
import multiprocessing
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import namedtuple

import common.parsing_utils as parsing

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class D2VWrapper:
    # content_categories might be set from outer scope
    # TODO: might be inferred from provided training content
    content_categories = ["eap", "fuse"]
    content_basepath = None
    basepath_suffix = "_content.csv"
    all_content_tagged_docs = None
    docs_category_mapping = None

    def __init__(self, content_basepath, basepath_suffix="_content.csv", content_categories=None, vector_length=300):
        # TODO: might as well try concatenation of multiple models
        #
        self.base_doc2vec_model = doc2vec.Doc2Vec(dm=0, size=vector_length, negative=12, hs=0, min_count=5,
                                                  workers=multiprocessing.cpu_count(), alpha=0.1)

        self.content_basepath = content_basepath
        self.basepath_suffix = basepath_suffix
        if content_categories:
            self.content_categories = content_categories

    def init_model_vocab(self):
        # initializes the vocabulary by the given categories (content_categories)
        # in the given directory (content_basepath)
        assert doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

        all_content_df = self.get_content_as_dataframe()

        # fills up the mapping of document ids (index) to its original categories
        # enabling the reverse search of docs vectors for each category and subsequent classification
        self.docs_category_mapping = pd.Series(data=all_content_df["target"])

        # selects a text of the most relevant attributes filled for each item of dataframe
        # (in format title, desc, plaintext_content)
        all_content_sens, sen_doc_mapping = parsing.select_training_content(all_content_df, make_document_mapping=True, sent_split=False)
        # transform the training sentences into TaggedDocument list
        self.all_content_tagged_docs = parsing.tagged_docs_from_content(all_content_sens, self.docs_category_mapping)

        self.base_doc2vec_model.build_vocab(self.all_content_tagged_docs)

    def train_model(self, shuffle=True, epochs=10):
        if self.all_content_tagged_docs is None:
            self.init_model_vocab()
        for epoch in range(epochs):
            logging.info("Training D2V model %s" % self.base_doc2vec_model)
            logging.info("Epoch %s convergence descent alpha: %s" % (epoch, self.base_doc2vec_model.alpha))
            if shuffle:
                train_ordered_tagged_docs = deepcopy(self.all_content_tagged_docs)
                random.shuffle(train_ordered_tagged_docs)
            else:
                train_ordered_tagged_docs = self.all_content_tagged_docs
            # self.base_doc2vec_model.infer_vector(self.base_doc2vec_model.vocab.keys()[:50][0:10])
            # vectors are inferable - might be just random init tho

            self.base_doc2vec_model.train(train_ordered_tagged_docs)

    def infer_content_vectors(self, category=None, infer_alpha=0.05, infer_subsample=0.05, infer_steps=10):
        # TODO: for testing purposes - remove
        limit = self.all_content_tagged_docs.__len__()
        if category is None:
            doc_vectors = [self.base_doc2vec_model.infer_vector(doc.words, infer_alpha, infer_subsample, infer_steps)
                           for doc in self.all_content_tagged_docs[:limit]]
            doc_categories = [doc.category_expected for doc in self.all_content_tagged_docs[:limit]]
            docs_vectors_df = pd.DataFrame(data=doc_vectors)
            docs_vectors_df["y"] = doc_categories

            # TODO: could try docs_vectors_df shuffling for a better performance of superior classifier
            return docs_vectors_df
        else:
            # TODO: implement if needed
            # returns vectors of only particular category
            return

    def get_content_as_dataframe(self, cat_label=None):
        if not cat_label:
            # retrieve all content of all known categories
            all_content = pd.DataFrame(
                columns="sys_title,sys_description,source,sys_content_plaintext,target".split(","))
            for cat_label in self.content_categories:
                new_content = pd.read_csv("%s/%s%s" % (self.content_basepath, cat_label, self.basepath_suffix),
                                          na_filter=False, error_bad_lines=False)
                all_content = all_content.append(new_content, ignore_index=True)
            return all_content
        else:
            # retrieve only one cat_label content
            cat_content = pd.read_csv("%s/%s%s" % (self.content_basepath, cat_label, self.basepath_suffix),
                                      na_filter=False)
            return cat_content

    # the docs must have been seen on training
    # the classification prediction is left on adjacent d2v_wrapper taking docs vectors as input
    def get_tagged_docs_vectors(self, tagged_docs):
        return [self.base_doc2vec_model.docvecs[doc.tags[0]] for doc in tagged_docs]
