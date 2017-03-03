# inspired by
# http://scikit-learn.org/stable/modules/cross_validation.html

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from model.w2v_classifier import W2VClassifier
from model.kfold_classifier_emulator import KFoldClassifierEmulator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# initialize d2v_wrapper providing as well metadata about the models state
classifier = W2VClassifier(content_basepath="../data/content/playground", basepath_suffix="_content.csv")
# select the categories to train on and classify
classifier.content_categories = ["eap", "fuse", "devstudio"]

# TODO: consider category samples for performance tweak -
cat_samples_limit = 50000
content_df = pd.DataFrame(columns=["content", "y", "doc_id"])

k_fold_splits = 5


def accuracy_for_category(y_expected, y_actual, label):
    label_expected = y_expected[y_expected == label]
    intersect = y_expected[np.where(y_expected == y_actual)]
    label_intersect = intersect[intersect == label]

    return float(len(label_intersect))/len(label_expected)

emulator = KFoldClassifierEmulator(classifier=classifier, splits=k_fold_splits)
emulator.gather_training_content()
for y_expected, y_actual in emulator.split_and_emulate(returns="classified"):
    print "Split accuracy: %s" % accuracy_score(y_expected, y_actual)
    print "Split mean accuracies for groups:"
    print classifier.content_categories
    # categories accuracies
    new_accuracies = map(lambda cat_label: accuracy_for_category(y_expected, y_actual, cat_label),
                         classifier.content_categories)
    print new_accuracies

print "done"
