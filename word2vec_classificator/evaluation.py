# inspired by
# http://scikit-learn.org/stable/modules/cross_validation.html

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from model.classifier import W2VClassifier
from model.common import parsing_utils as parsing

from copy import deepcopy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

label_content_mapping = {"portal": "../data/content/portal_content.csv"}

# initialize classifier providing as well metadata about the models state
classifier = W2VClassifier(content_basepath="../data/content/", basepath_suffix="_content.csv")

# TODO: consider category samples for performance tweak -
cat_samples_limit = 5000
content_df = pd.DataFrame(columns=["content", "y", "doc_id"])

k_fold_splits = 10

# content aggregation into data frame
for sample_cat in classifier.content_categories:

    cat_content_df = classifier.get_content_as_dataframe(cat_label=sample_cat)
    cat_content_df = cat_content_df[:cat_samples_limit]
    logging.info("Loading %s documents of category '%s'" % (len(cat_content_df), sample_cat))

    # cat_size = cat_size if len(cat_content_df) >= cat_size else len(cat_content_df)

    # cat_content_sample = np.random.choice(cat_content_df, size=cat_size, replace=False)
    cat_content, document_mapping = parsing.select_training_content(cat_content_df, make_document_mapping=True)
    new_data_df = pd.DataFrame(data=cat_content, columns=["content"])
    new_data_df["y"] = np.array([sample_cat] * len(new_data_df))
    new_data_df["doc_id"] = pd.Series(document_mapping)
    new_data_df["doc_id"] = new_data_df.apply(lambda entry: "%s:%s" % (entry["y"], str(entry["doc_id"])), axis=1)
    content_df = content_df.append(new_data_df)

logging.info("Loaded %s sentences of %s categories" % (len(content_df), len(classifier.content_categories)))

# categories' contents size balancing in train/test split is left for stratified k-fold algorithm
strat_kfold = StratifiedKFold(n_splits=k_fold_splits, shuffle=True)
accuracies = []
cat_accuracies = pd.DataFrame(columns=classifier.content_categories)


def accuracy_for_category(y_expected, y_actual, label):
    label_expected = y_expected[y_expected == label]
    intersect = y_expected[np.where(y_expected == y_actual)]
    label_intersect = intersect[intersect == label]

    return float(len(label_intersect))/len(label_expected)

classifier.init_vocab_model()

# used for document-content mapping
all_docs_ids = pd.Series(content_df["doc_id"].unique())
xy_split = pd.DataFrame()
xy_split["doc_id"] = all_docs_ids
xy_split["y"] = all_docs_ids.apply(lambda doc_id: doc_id.split(":")[0])

for train_doc_indices, test_doc_indices in strat_kfold.split(xy_split["doc_id"], xy_split["y"]):
    # training part
    train_docs = xy_split.iloc[train_doc_indices]["doc_id"]
    training_content = content_df[content_df["doc_id"].isin(train_docs)]
    # train each category model with its content

    # TODO: consider removal
    # balance training set so that all categories are trained on same big content
    training_size = training_content.groupby(by="y").size().min()

    for cat_label in classifier.content_categories:
        classifier.init_model_from_dataframe(cat_label=cat_label,
                                             cat_sentences_df=(training_content[training_content["y"] == cat_label]
                                                               ["content"][:training_size]))

    # testing part
    test_df = xy_split.iloc[test_doc_indices]
    test_df["sentences"] = test_df["doc_id"].apply(lambda doc: parsing.sentence_list_from_content(
        content_df[content_df["doc_id"] == doc]))

    y_expected = test_df["y"].values

    xy_actual = classifier.predict_all(test_df["sentences"])
    y_actual = xy_actual["y"]
    split_accuracy = accuracy_score(y_expected, y_actual)
    accuracies.append(split_accuracy)
    print "split accuracy: %s" % accuracies[-1]

    # TODO: accuracy of each group after each run
    new_accuracies = map(lambda cat_label: accuracy_for_category(y_expected, y_actual, cat_label), classifier.content_categories)

    cat_accuracies = cat_accuracies.append(pd.DataFrame(data=[new_accuracies], columns=classifier.content_categories))

# sample_sentences = all_sentences.ix[15]
print "%s splits accuracies: %s" % (len(accuracies), accuracies)
print "mean accuracy: %s" % np.array(accuracies).mean()
print "mean accuracies for groups:"
print cat_accuracies.mean()
print "done"
