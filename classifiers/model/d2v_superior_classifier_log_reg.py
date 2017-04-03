# inspired by
# http://scikit-learn.org/stable/modules/cross_validation.html

import logging

import numpy as np
import pandas as pd
# tried models:
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from doc2vec_wrapper import D2VWrapper

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEST_MODE = True

# initialize d2v_wrapper providing also metadata about the models state
d2v_wrapper = D2VWrapper(content_basepath="../../data/content/playground/auto",
                         basepath_suffix="_content.csv",
                         content_categories=['amq', 'webserver', 'datagrid'])
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
    if len(label_expected) == 0:
        logging.warn("Accuracy of %s category evaluated on 0 samples" % label)
        return 1 if len(label_intersect) == 0 else 0
    else:
        return float(len(label_intersect)) / len(label_expected)

# classifier training and eval:
strat_kfold = StratifiedKFold(n_splits=3, shuffle=True)
accuracies = []
cat_accuracies = pd.DataFrame(columns=d2v_wrapper.content_categories)

for train_doc_indices, test_doc_indices in strat_kfold.split(doc_vectors, doc_labels):
    # training
    log_reg_classifier = LogisticRegression(solver="newton-cg", multi_class='ovr', n_jobs=8)
    log_reg_classifier.fit(doc_vectors.iloc[train_doc_indices], doc_labels.iloc[train_doc_indices])

    # testing
    y_expected = doc_labels.iloc[test_doc_indices].values
    y_actual = log_reg_classifier.predict(doc_vectors.iloc[test_doc_indices])

    # evaluation
    split_accuracy = accuracy_score(y_expected, y_actual)
    logging.info("Run accuracy: %s" % split_accuracy)
    accuracies.append(split_accuracy)
    split_cat_accuracies = map(lambda cat: accuracy_for_category(y_expected, y_actual, cat), d2v_wrapper.content_categories)
    logging.info("Cat accuracies:\n%s" % split_cat_accuracies)
    cat_accuracies = cat_accuracies.append(pd.DataFrame(data=[split_cat_accuracies], columns=d2v_wrapper.content_categories))

print "done"
print "accuracies: %s" % accuracies
print "mean accuracy: %s" % np.mean(accuracies)
print "categories accuracies: \n%s" % cat_accuracies.mean()
print "done"
