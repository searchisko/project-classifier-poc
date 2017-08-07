# inspired by
# https://deeplearning4j.org/word2vec.html
# https://deeplearning4j.org/welldressed-recommendation-engine
# http://scikit-learn.org/stable/modules/cross_validation.html

# used for debugging and POC - functional demonstration in same-named jupyter notebook
from __future__ import print_function

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from analyses.lab.dependencies import scores_tuner
from doc2vec_wrapper import D2VWrapper
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)

TEST_MODE = False

# initialize d2v_wrapper providing also metadata about the models state
content_categories = ["amq", "eap", "webserver", "datagrid", "fuse", "brms", "bpmsuite", "devstudio", "cdk",
                "developertoolset", "rhel", "softwarecollections", "mobileplatform", "openshift"]

# initialize d2v_wrapper providing as well metadata about the models state
d2v_wrapper = D2VWrapper(content_categories=content_categories,
                         vector_length=500)

# EITHER initialize the vocab of documents and minimize the distances of embeddings in training phase
# d2v_wrapper.init_model_vocab(content_basepath="../../data/content/playground/auto/nostem",
#                              basepath_suffix="_content.csv", drop_short_docs=10)
# d2v_wrapper.train_model(shuffle=True, epochs=1 if TEST_MODE else 15)
#
# d2v_wrapper.persist_trained_wrapper("trained_models/wrapper/")

# OR load initialized and trained wrapper if available
d2v_wrapper.load_persisted_wrapper("trained_models/wrapper/header_incl/10epoch_train_stem_not_removed_header_v400")

doc_vectors_labeled = d2v_wrapper.infer_vocab_content_vectors()

doc_vectors = doc_vectors_labeled.iloc[:, :-1]
doc_labels = doc_vectors_labeled.iloc[:, -1]

# softwarecollections is too small for CV fold tuning, so we'll exclude it
limit_prod_list = ["softwarecollections"]
doc_vectors = doc_vectors[~doc_labels.isin(limit_prod_list)]
doc_labels = doc_labels[~doc_labels.isin(limit_prod_list)]

# TODO: content limited to given categories
# limit the content to specific categories
# limit_prod_list = ["webserver", "datagrid", "fuse", "brms"]
# doc_vectors = doc_vectors[doc_labels.isin(limit_prod_list)]
# doc_labels = doc_labels[doc_labels.isin(limit_prod_list)]


# extended evaluation metric on selected category
def accuracy_for_category(y_expected, y_actual, label):
    label_expected = y_expected[y_expected == label]
    intersect = y_expected[np.where(y_expected == y_actual)]
    label_intersect = intersect[intersect == label]
    if len(label_expected) == 0:
        logging.warn("Accuracy of %s category evaluated on 0 samples" % label)
        return 1 if len(label_intersect) == 0 else 0
    else:
        return float(len(label_intersect)) / len(label_expected)

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


# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    doc_vectors, doc_labels, test_size=0.2, random_state=0)


# Set the parameters by cross-validation
def frange(x, y, jump):
    out = []
    while x < y:
        out.append(x)
        x += jump
    return out

tuned_parameters = {'C': frange(0.2, 0.3, 0.01)}

print("# Tuning hyper-parameters for combined performance metric maximization")
print()


# evaluation wrapper for given estimator, to conform with sklearn's GridSearchCV interface
class LogRegCustomScore(LogisticRegression):
    def score(self, X, y, _=None):
        predicted_probs_df = pd.DataFrame(data=self.predict_proba(X), columns=list(self.classes_), index=X.index)
        reached_score = scores_tuner.evaluate(y, predicted_probs_df)
        print("LogReg on C: %s performance: %s" % (self.C, reached_score))
        return reached_score


clf = GridSearchCV(LogRegCustomScore(solver="sag", multi_class='ovr', n_jobs=8, max_iter=1000),
                   tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.5f (+/-%0.05f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()
logging.info("Done")