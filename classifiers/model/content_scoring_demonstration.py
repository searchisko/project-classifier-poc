# inspired by https://github.com/TaddyLab/gensim/blob/deepir/docs/notebooks/deepir.ipynb
# no further maintained file - concept has been merged to evaluation.py

from gensim.models import Word2Vec
import multiprocessing
import pandas as pd
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

label_content_mapping = {"fsw": "../../data/content/other/fsw_content.csv"}


# split tokenized text into all_sentences
def sentence_split(document):
    return filter(lambda sentence: len(sentence) > 0, document.split("."))


# split tokenized sentence (sequence of tokens separated by space) into tokens
def token_split(sentence):
    return filter(lambda token: len(token) > 0, sentence.split(" "))


def get_labeled_content(df):
    training_attributes = ["sys_content_plaintext", "sys_description", "sys_title"]
    # df = pd.read_csv(label_content_mapping[label], na_filter=False)

    sentence_container = np.array([])

    # considers optionally sys_content_plaintext if filled, or sys_description if not
    training_text_series = df.apply(lambda content: sentence_split(content[training_attributes[0]])
                                    if content[training_attributes[0]] else
                                        sentence_split(content[training_attributes[1]]) if content[training_attributes[1]]
                                            else content[training_attributes[2]],
                                    axis=1)
    for sentences in training_text_series:
        sentence_container = np.append(sentence_container, sentences)

    # TODO: performance bottleneck supposedly here
    sentence_container = pd.Series(map(lambda sentence: token_split(sentence), sentence_container))

    return sentence_container


df = pd.read_csv(label_content_mapping["fsw"], na_filter=False)

all_sentences = get_labeled_content(df)
base_model = Word2Vec(
    workers=multiprocessing.cpu_count(),
    iter=30, # iter = sweeps of SGD through the data; more is better
    hs=1, negative=0 # we only have scoring for the hierarchical softmax setup
    )

# TODO: build vocab on total train sample
base_model.build_vocab(all_sentences)

simple_train_test_split = 0.8

train_test_split_index = int(len(all_sentences) * simple_train_test_split)
train_data = all_sentences.iloc[:train_test_split_index]

# TODO: train model with only vocab of the given category
from copy import deepcopy
category_model = deepcopy(base_model)

category_model.train(train_data, total_examples=len(train_data))

print category_model
print category_model.vocab

i = train_test_split_index

test_sentences = all_sentences[train_test_split_index:]
#
# sentences_log_probs = np.array([category_model.score(sen, sen) for sen in test_sentences])
# sentence_probs = np.exp(sentences_log_probs)

# for sen_prob in sentence_probs:
#     print "sentence %s score: %s" % (i, sen_prob)
#     i += 1
max_score = -100000
for sen in test_sentences:
    # sens = all_sentences.iloc[i]
    sen_log_p = category_model.score([sen], len(sen))
    if sen_log_p > max_score:
        max_score = sen_log_p

sens_p = []
for sen in test_sentences:
    # sens = all_sentences.iloc[i]
    sen_log_p = category_model.score([sen], len(sen))
    sen_p = np.exp(sen_log_p - max_score)
    sens_p.append(sen_p)
    print "sentence i score: %s" % (sen_log_p)
    print "sentence i exp score: %s" % (sen_p)

mn = np.mean(sens_p)
sens_p = map(lambda p: p/mn, sens_p)

# sample_sentences = all_sentences.ix[15]
print "done"