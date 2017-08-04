# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
# https://groups.google.com/forum/#!msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ
import cPickle
import logging
import random
from copy import deepcopy

from os import listdir
from os.path import isfile, join

import pandas as pd
from gensim.models import doc2vec
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import parsing_utils as parsing

# parallel support
import joblib
from joblib import Parallel, delayed
import multiprocessing
# from .categorized_document import CategorizedDocument
# from collections import namedtuple
# CategorizedDocument = namedtuple('CategorizedDocument', 'words tags category_expected header_words')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def async_trigger(wrapper, wordlist):
    return wrapper.infer_content_vector(wordlist)


class D2VWrapper:
    # content_categories might be set from outer scope
    train_content_tagged_docs = None
    docs_category_mapping = None
    content_categories = None
    inferred_vectors = None
    header_docs = None

    def __init__(self, content_categories=None, vector_length=300, window=8, train_algo="dbow"):
        self.base_doc2vec_model = doc2vec.Doc2Vec(dm=1 if train_algo == "dm" else 0, size=vector_length, negative=12,
                                                  hs=0, min_count=5, workers=multiprocessing.cpu_count(),
                                                  alpha=0.1, window=window)

        if content_categories:
            self.content_categories = content_categories

    def init_model_vocab(self, content_basepath, basepath_suffix="_content.csv", drop_short_docs=False):
        # infer the trained categories according to the files in directory
        dir_files = [f for f in listdir(content_basepath) if isfile(join(content_basepath, f))]
        self.content_categories = map(
            lambda dir_file_path: dir_file_path.replace(content_basepath, "").replace(basepath_suffix, ""), dir_files)

        # initializes the vocabulary by the given categories (content_categories)
        # in the given directory (content_basepath)
        assert doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

        all_content_df = parsing.get_content_as_dataframe(content_basepath, basepath_suffix, self.content_categories)

        # fills up the mapping of document ids (index) to its original categories
        # enabling the reverse search of all_base_vocab_docs vectors for each category and subsequent classification
        self.docs_category_mapping = pd.Series(data=all_content_df["target"])

        # selects a text of the most relevant attributes filled for each item of dataframe
        # (in format title, desc, plaintext_content)
        all_content_sens, _ = parsing.select_training_content(all_content_df, make_document_mapping=True, sent_split=False)
        all_content_headers = parsing.select_headers(all_content_df)

        logging.info("Loaded %s all_base_vocab_docs from %s categories" % (len(all_content_sens), len(self.content_categories)))

        # filter short all_base_vocab_docs from training, if required
        if drop_short_docs:
            logging.info("Filtering all_base_vocab_docs shorter than %s tokens from vocab sample" % drop_short_docs)

            content_lens = all_content_sens.apply(lambda content: len(content))
            ok_indices = content_lens >= drop_short_docs

            all_content_sens = all_content_sens[ok_indices]
            self.docs_category_mapping = self.docs_category_mapping[ok_indices.values]

            logging.info("%s all_base_vocab_docs included in vocab init" % self.docs_category_mapping.__len__())

        # transform the training sentences into TaggedDocument list
        self.train_content_tagged_docs = parsing.tagged_docs_from_content(all_content_sens,
                                                                          all_content_headers,
                                                                          self.docs_category_mapping)
        self.train_content_tagged_docs = self.train_content_tagged_docs.reset_index(drop=True)

        self.init_vocab_from_docs()

    def init_vocab_from_docs(self, docs=None, deduplicate=True):
        if docs is not None:
            logging.info("Initializing d2v vocab from externally parsed CategorizedDocs")
            self.train_content_tagged_docs = docs

        if deduplicate:
            logging.info("De-duplicating training content docs")
            self.train_content_tagged_docs = parsing.drop_duplicate_docs(self.train_content_tagged_docs)

        # derive training all_base_vocab_docs of header content and push it into model vocabulary
        self.header_docs = parsing.parse_header_docs(self.train_content_tagged_docs)

        self.base_doc2vec_model.build_vocab(self.train_content_tagged_docs.append(self.header_docs))

        # after this step, vector for any list of words is inferable - though the docs vectors needs to be embedded in
        # train_model() all_base_vocab_docs vectors should be retrieved first after the training

    def train_model(self, shuffle=True, epochs=10):
        # now training on headers as well

        if self.train_content_tagged_docs is None:
            logging.error("D2V vocabulary not initialized. Training must follow the init_model_vocab()")
            return
        for epoch in range(epochs):
            logging.info("Training D2V model %s" % self.base_doc2vec_model)
            logging.info("Epoch %s convergence descent alpha: %s" % (epoch, self.base_doc2vec_model.alpha))

            # shuffle support
            train_ordered_tagged_docs = deepcopy(self.train_content_tagged_docs.values)
            train_ordered_headers = deepcopy(self.header_docs.values)
            if shuffle and epoch > 0:
                # shuffling is time-consuming and is not necessary in the first epoch (current order not seen before)
                random.shuffle(train_ordered_tagged_docs)
                random.shuffle(train_ordered_headers)
            else:
                train_ordered_tagged_docs = self.train_content_tagged_docs
            # self.base_doc2vec_model.infer_vector(self.base_doc2vec_model.vocab.keys()[:50][0:10])

            self.base_doc2vec_model.train(pd.Series(train_ordered_tagged_docs).append(pd.Series(train_ordered_headers)))
            # self.base_doc2vec_model.train(train_ordered_headers)

    def persist_trained_wrapper(self, model_save_dir, model_only=False):
        # if persisting folder does not exist, create it - Service layer will take care of it
        if not model_only:
            logging.info("Serializing wrapper model to: %s" % model_save_dir)

            logging.info("Persisting all_base_vocab_docs objects")
            joblib.dump(self.train_content_tagged_docs, model_save_dir + "/doc_labeling.mod")

            logging.info("Persisting inferred vectors")
            joblib.dump(self.inferred_vectors, model_save_dir + "/doc_vectors.mod")

        logging.info("Persisting trained Doc2Vec model")
        self.base_doc2vec_model.save(model_save_dir + "/doc2vec.mod")

    def load_persisted_wrapper(self, model_save_dir, model_only=False):
        logging.info("Loading serialized wrapper model from: %s" % model_save_dir)

        if not model_only:
            logging.info("Loading all_base_vocab_docs objects")
            self.train_content_tagged_docs = joblib.load(model_save_dir + "/doc_labeling.mod")

            self.content_categories = self.train_content_tagged_docs.apply(lambda doc: doc.category_expected).unique()

            # header content parse from base all_base_vocab_docs objects
            self.header_docs = parsing.parse_header_docs(self.train_content_tagged_docs)

            logging.info("Loading all_base_vocab_docs vectors")
            self.inferred_vectors = joblib.load(model_save_dir + "/doc_vectors.mod")

        logging.info("Loading trained Doc2Vec model")
        self.base_doc2vec_model = doc2vec.Doc2Vec.load(model_save_dir + "/doc2vec.mod")

    def infer_vocab_content_vectors(self, new_inference=False, category=None):
        if not new_inference:
            if self.inferred_vectors is not None:
                logging.info("Returning already inferred doc vectors of %s all_base_vocab_docs" % len(self.inferred_vectors))
                return self.inferred_vectors

        logging.info("Docs vector inference started")
        if category is None:
            # inference with default aprams config
            # TODO: try other inference params on new inference
            self.inferred_vectors = self.infer_content_vectors(self.train_content_tagged_docs)

            self.inferred_vectors["y"] = [doc.category_expected for doc in self.train_content_tagged_docs]

            return self.inferred_vectors
        else:
            # returns vectors of only a particular category
            # implement if needed
            return

    def infer_content_vector(self, wordlist, infer_alpha=0.05, infer_subsample=0.05, infer_steps=10):
        return self.base_doc2vec_model.infer_vector(wordlist, infer_alpha, infer_subsample, infer_steps)

    def infer_vectors_parallel(self, docs, jobs=multiprocessing.cpu_count()):
        pool = multiprocessing.Pool(processes=jobs)
        results = [pool.apply_async(async_trigger, args=(self, doc.words)) for doc in docs]
        results = [p.get() for p in results]

        return results
        # Parallel(n_jobs=jobs)(delayed(self._infer_content_vector)(doc.words) for doc in docs)

    def _infer_vectors_non_parallel(self, docs, infer_alpha=0.05, infer_subsample=0.05, infer_steps=10):
        return [self.base_doc2vec_model.infer_vector(doc.words, infer_alpha, infer_subsample, infer_steps) for doc in docs]

    # gets a pd.Series of CategorizedDocument-s with unfilled categories
    # returns a vectors matrix for a content of the input CategorizedDpcument-s in the same order
    def infer_content_vectors(self, docs, infer_cycles=10):
        # that might probably be tested on already classified data
        header_docs = parsing.parse_header_docs(docs)

        # TODO: parallelization by pooling is not sustainable - needs batching
        # content_vectors = self.infer_vectors_parallel(docs)

        # collect the docs vectors in <infer_cycles> inference repetitions and average the results
        doc_vectors = np.zeros((len(docs), self.base_doc2vec_model.vector_size*2, infer_cycles))
        for infer_i in range(infer_cycles):
            logging.info("Inferring vectors of %s documents in %s/%s cycle" % (len(docs), infer_i, infer_cycles))
            doc_vectors[:, :self.base_doc2vec_model.vector_size, infer_i] = self._infer_vectors_non_parallel(docs)
            # header vectors inference
            logging.info("Inferring vectors of %s headers in %s/%s cycle" % (len(header_docs), infer_i, infer_cycles))
            doc_vectors[:, self.base_doc2vec_model.vector_size:, infer_i] = self._infer_vectors_non_parallel(header_docs)

        # average the <infer_cycles> inferences
        doc_vectors_avgs = doc_vectors.mean(axis=2)

        doc_vectors_df = pd.DataFrame(doc_vectors_avgs)

        # rename vector columns incrementally - columns are required tu have unique id by NN classifier
        doc_vectors_df.columns = np.arange(len(doc_vectors_df.columns))

        return doc_vectors_df

    def get_doc_content(self, index, word_split=False):
        if word_split:
            return self.train_content_tagged_docs.iloc[index].words
        else:
            return parsing.content_from_words(self.train_content_tagged_docs.iloc[index].words)

