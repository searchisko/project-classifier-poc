# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
# https://groups.google.com/forum/#!msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ
import cPickle
import logging
import multiprocessing
import random
from copy import deepcopy

import pandas as pd
from gensim.models import doc2vec

import common.parsing_utils as parsing
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class D2VWrapper:
    # content_categories might be set from outer scope
    # TODO: might be inferred from provided training content
    content_categories = None
    all_content_tagged_docs = None
    docs_category_mapping = None
    inferred_vectors = None

    def __init__(self, content_categories=None, vector_length=300):
        # TODO: might as well try concatenation of multiple models
        #
        self.base_doc2vec_model = doc2vec.Doc2Vec(dm=0, size=vector_length, negative=12, hs=0, min_count=5,
                                                  workers=multiprocessing.cpu_count(), alpha=0.1)

        if content_categories:
            self.content_categories = content_categories

    def init_model_vocab(self, content_basepath, basepath_suffix, drop_short_docs=False):
        # initializes the vocabulary by the given categories (content_categories)
        # in the given directory (content_basepath)
        assert doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

        all_content_df = self.get_content_as_dataframe(content_basepath, basepath_suffix).drop_duplicates()

        # fills up the mapping of document ids (index) to its original categories
        # enabling the reverse search of docs vectors for each category and subsequent classification
        self.docs_category_mapping = pd.Series(data=all_content_df["target"])

        # selects a text of the most relevant attributes filled for each item of dataframe
        # (in format title, desc, plaintext_content)
        all_content_sens, _ = parsing.select_training_content(all_content_df, make_document_mapping=True, sent_split=False)

        logging.info("Loaded %s docs from %s categories" % (len(all_content_sens), len(self.content_categories)))

        # filter short docs from training, if required
        if drop_short_docs:
            logging.info("Filtering docs shorter than %s tokens from vocab sample" % drop_short_docs)

            content_lens = all_content_sens.apply(lambda content: len(content))
            ok_indices = content_lens >= drop_short_docs

            all_content_sens = all_content_sens[ok_indices]
            self.docs_category_mapping = self.docs_category_mapping[ok_indices.values]

            logging.info("%s docs included in vocab init" % self.docs_category_mapping.__len__())

        # transform the training sentences into TaggedDocument list
        self.all_content_tagged_docs = parsing.tagged_docs_from_content(all_content_sens, self.docs_category_mapping)
        self.all_content_tagged_docs = self.all_content_tagged_docs.reset_index(drop=True)

        self.base_doc2vec_model.build_vocab(self.all_content_tagged_docs)
        # after this step, vectors are already inferable - might be just a random init tho
        # docs vectors should be retrieved first after the training

    def train_model(self, trained_model_path=None, shuffle=True, epochs=10, save_model_path=None):
        if trained_model_path is not None:
            self.base_doc2vec_model = doc2vec.Doc2Vec.load(trained_model_path)
            return

        if self.all_content_tagged_docs is None:
            self.init_model_vocab()
        for epoch in range(epochs):
            logging.info("Training D2V model %s" % self.base_doc2vec_model)
            logging.info("Epoch %s convergence descent alpha: %s" % (epoch, self.base_doc2vec_model.alpha))
            if shuffle and epoch > 0:
                # shuffling is time-consuming and is not necessary in the first epoch (current order not seen before)
                train_ordered_tagged_docs = deepcopy(self.all_content_tagged_docs)
                random.shuffle(train_ordered_tagged_docs)
            else:
                train_ordered_tagged_docs = self.all_content_tagged_docs
            # self.base_doc2vec_model.infer_vector(self.base_doc2vec_model.vocab.keys()[:50][0:10])

            self.base_doc2vec_model.train(train_ordered_tagged_docs)

        if save_model_path is not None:
            self.base_doc2vec_model.save(save_model_path)

    def persist_trained_wrapper(self, model_save_path):
        logging.info("Serializing wrapper model to: %s" % model_save_path)

        logging.info("Persisting docs objects")
        with open(model_save_path+"doc_labeling.mod", "w") as pickle_file_writer:
            cPickle.dump(self.all_content_tagged_docs, pickle_file_writer)

        logging.info("Persisting inferred vectors")
        with open(model_save_path+"doc_vectors.mod", "w") as pickle_file_writer:
            cPickle.dump(self.inferred_vectors, pickle_file_writer)

        logging.info("Persisting trained Doc2Vec model")
        self.base_doc2vec_model.save(model_save_path + "doc2vec.mod")

    def load_persisted_wrapper(self, model_save_path):
        logging.info("Loading serialized wrapper model from: %s" % model_save_path)

        logging.info("Loading docs objects")
        with open(model_save_path + "doc_labeling.mod", "r") as pickle_file_reader:
            self.all_content_tagged_docs = cPickle.load(pickle_file_reader)

        logging.info("Loading docs vectors")
        with open(model_save_path + "doc_vectors.mod", "r") as pickle_file_reader:
            self.inferred_vectors = cPickle.load(pickle_file_reader)

        logging.info("Loading trained Doc2Vec model")
        self.base_doc2vec_model = doc2vec.Doc2Vec.load(model_save_path + "doc2vec.mod")

    def infer_content_vectors(self, new_inference=False, category=None, infer_alpha=0.05, infer_subsample=0.05, infer_steps=10):
        # TODO: try other inference params

        if not new_inference:
            if self.inferred_vectors is not None:
                logging.info("Returning already inferred doc vectors")
                return self.inferred_vectors
        else:
            logging.warn("Vector inference is being repeated now.")

        logging.info("Docs vector inference started")
        if category is None:
            logging.info("Inferring vectors of %s documents" % len(self.all_content_tagged_docs))

            doc_vectors = [self.base_doc2vec_model.infer_vector(doc.words, infer_alpha, infer_subsample, infer_steps)
                           for doc in self.all_content_tagged_docs]
            doc_categories = [doc.category_expected for doc in self.all_content_tagged_docs]
            self.inferred_vectors = pd.DataFrame(data=doc_vectors)
            self.inferred_vectors["y"] = doc_categories

            return self.inferred_vectors
        else:
            # returns vectors of only a particular category
            # implement if needed
            return

    def get_content_as_dataframe(self, content_basepath, basepath_suffix, cat_label=None):
        if not cat_label:
            # retrieve all content of all known categories
            all_content = pd.DataFrame(
                columns="sys_title,sys_description,source,sys_content_plaintext,target".split(","))
            for cat_label in self.content_categories:
                new_content = pd.read_csv("%s/%s%s" % (content_basepath, cat_label, basepath_suffix),
                                          na_filter=False, error_bad_lines=False)
                all_content = all_content.append(new_content, ignore_index=True)
            return all_content
        else:
            # retrieve only one cat_label content
            cat_content = pd.read_csv("%s/%s%s" % (content_basepath, cat_label, basepath_suffix),
                                      na_filter=False)
            return cat_content

    def get_doc_content(self, index, word_split=False):
        if word_split:
            return self.all_content_tagged_docs.iloc[index].words
        else:
            return parsing.content_from_words(self.all_content_tagged_docs.iloc[index].words)






