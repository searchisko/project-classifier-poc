# inspired by
# https://deeplearning4j.org/word2vec.html
# https://deeplearning4j.org/welldressed-recommendation-engine
# https://www.tensorflow.org/get_started/tflearn
# http://scikit-learn.org/stable/modules/cross_validation.html

# used for debugging  and POC - functional demonstration in same-named jupyter notebook

import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from doc2vec_wrapper import D2VWrapper

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.level

TEST_MODE = True

# initialize d2v_wrapper providing also metadata about the models state
d2v_wrapper = D2VWrapper(content_basepath="../../data/content/playground",
                         basepath_suffix="_content.csv",
                         content_categories=["eap", "fuse", "devstudio"])
# select the categories to train on and classify


d2v_wrapper.init_model_vocab()
d2v_wrapper.train_model(shuffle=True, epochs=1 if TEST_MODE else 10)

doc_vectors_labeled = d2v_wrapper.infer_content_vectors()
doc_vectors = doc_vectors_labeled.iloc[:, :-1]
doc_labels = doc_vectors_labeled.iloc[:, -1]


def accuracy_for_category(y_expected, y_actual, label):
    label_expected = y_expected[y_expected == label]
    intersect = y_expected[np.where(y_expected == y_actual)]
    label_intersect = intersect[intersect == label]

    return float(len(label_intersect)) / len(label_expected)


# tensorflow new version estimator support
def input_fn(data_set, dummified_target):
    feature_cols = {str(k): tf.constant(data_set.iloc[k].values, dtype=tf.float32)
                    for k in range(data_set.shape[1])}
    labels = tf.constant(dummified_target)
    return feature_cols, labels

# encoding/decoding target categories
mapping = []


def encode_categories(target_series):
    global mapping
    if not len(mapping):
        for cat in target_series.unique():
            mapping.append(cat)

    return target_series.apply(lambda cat_str: mapping.index(cat_str))


def decode_categories(target_series):
    global mapping
    return target_series.apply(lambda cat_idx: mapping[cat_idx])


def dataset_from_dataframe(data_df, target_series):
    dataset = tf.contrib.learn.datasets.base.Dataset(data=data_df.values, target=encode_categories(target_series))
    return dataset

# classifier training and eval:
strat_kfold = StratifiedKFold(n_splits=3, shuffle=True)
accuracies = []
cat_accuracies = pd.DataFrame(columns=d2v_wrapper.content_categories)

for train_doc_indices, test_doc_indices in strat_kfold.split(doc_vectors, doc_labels):

    random.shuffle(train_doc_indices)
    random.shuffle(test_doc_indices)
    # init model
    # model consists of two layers
    # 1. dense, fully-connected layer with relu act. function
    # 2. softmax output layer for classificatio and scoring
    # train_vectors = tf.constant(doc_vectors.iloc[train_doc_indices].values)
    train_vectors = doc_vectors.iloc[train_doc_indices]

    # train_vectors_t = tf.constant(train_vectors, dtype=tf.float64, name="vectors")

    y_true = doc_labels.iloc[train_doc_indices]
    # revertable by idxmax(axis=1)
    y_true_onehot = pd.get_dummies(y_true).values

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=doc_vectors.shape[1])]
    tf_dataset = dataset_from_dataframe(train_vectors, y_true)

    two_layer_nn_classifier = SKCompat(tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                                      hidden_units=[doc_vectors.shape[1] / 2],
                                                                      activation_fn=tf.nn.relu,
                                                                      dropout=0.1,
                                                                      n_classes=len(d2v_wrapper.content_categories),
                                                                      optimizer="Adam"))
    # two_layer_nn_classifier.fit(input_fn=lambda: input_fn(train_vectors, y_true_onehot))
    two_layer_nn_classifier.fit(x=tf_dataset.data,
                                y=tf_dataset.target,
                                steps=2000)
    # y_actual = two_layer_nn_classifier.predict(doc_vectors.iloc[test_doc_indices])
    #
    logits = two_layer_nn_classifier.predict(doc_vectors.iloc[test_doc_indices])
    y_actual = decode_categories(pd.Series(logits["classes"])).values

    # testing
    y_expected = doc_labels.iloc[test_doc_indices].values
    # y_actual = log_reg_classifier.predict(doc_vectors.iloc[test_doc_indices])

    # evaluation
    # accuracy can be simply computed with
    # two_layer_nn_classifier.score(x=doc_vectors.iloc[test_doc_indices],
    #                               y=encode_categories(doc_labels.iloc[test_doc_indices]))

    split_accuracy = accuracy_score(y_expected, y_actual)
    logging.info("Run accuracy: %s" % split_accuracy)
    accuracies.append(split_accuracy)
    split_cat_accuracies = map(lambda cat: accuracy_for_category(y_expected, y_actual, cat),
                               d2v_wrapper.content_categories)
    logging.info("Cat accuracies:\n%s" % split_cat_accuracies)
    cat_accuracies = cat_accuracies.append(
        pd.DataFrame(data=[split_cat_accuracies], columns=d2v_wrapper.content_categories))

print("done")
print("accuracies: %s" % accuracies)
print("mean accuracy: %s" % np.mean(accuracies))
print("categories accuracies: \n%s" % cat_accuracies.mean())
print("done")
