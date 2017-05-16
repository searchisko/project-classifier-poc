# copy of d2v_superior_classifier_log_reg.py

# inspired by
# http://scikit-learn.org/stable/modules/cross_validation.html

import logging
from copy import deepcopy
import multiprocessing

import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from classifiers.model.doc2vec_wrapper import D2VWrapper

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# optimization steps are reduced on TEST_MODE
TEST_MODE = False

# target categories
product_list = ["amq", "eap", "webserver", "datagrid", "fuse", "brms", "bpmsuite", "devstudio", "cdk",
                "developertoolset", "rhel", "softwarecollections", "mobileplatform", "openshift"]

# product_list = ["webserver", "datagrid", "amq"]

# initialize d2v_wrapper providing as well metadata about the models state
d2v_wrapper_base = D2VWrapper(content_categories=product_list,
                              vector_length=500)

# EITHER initialize the vocab of documents and minimize the distances of embeddings in training phase
d2v_wrapper_base.init_model_vocab(content_basepath="data/content/prod",
                                  basepath_suffix="_content.csv", drop_short_docs=10)

# d2v_wrapper.persist_trained_wrapper(model_save_path="trained_models/wrapper/10epoch_train_stem_not_removed_header")

# OR load initialized and trained wrapper if available
# d2v_wrapper.load_persisted_wrapper("trained_models/wrapper/10epoch_train_stem_not_removed_header_v400")


def accuracy_for_category(y_expected, y_actual, label):
    label_expected = y_expected[y_expected == label]
    intersect = y_expected[np.where(y_expected == y_actual)]
    label_intersect = intersect[intersect == label]
    if len(label_expected) == 0:
        logging.warn("Accuracy of %s category evaluated on 0 samples" % label)
        return 1 if len(label_intersect) == 0 else 0
    else:
        return float(len(label_intersect)) / len(label_expected)


# results collection
accuracies = []
cat_accuracies = pd.DataFrame(columns=d2v_wrapper_base.content_categories)
logits = pd.DataFrame(columns=["actual_prob", "expected_prob", "actual_class", "expected_class"])
wrong_docs_ids = pd.Series()

all_base_vocab_docs = d2v_wrapper_base.all_content_tagged_docs
doc_labels = all_base_vocab_docs.apply(lambda doc: doc.category_expected)

# evaluation on CV split persisting the categories respective size on each split
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True)

for train_doc_indices, test_doc_indices in strat_kfold.split(all_base_vocab_docs, doc_labels):

    random.shuffle(train_doc_indices)
    random.shuffle(test_doc_indices)

    train_docs = all_base_vocab_docs[train_doc_indices]
    test_docs = all_base_vocab_docs[test_doc_indices]

    d2v_wrapper_fold = D2VWrapper(content_categories=d2v_wrapper_base.content_categories, vector_length=500)
    d2v_wrapper_fold.init_vocab_from_docs(train_docs)

    # docs vectors embedding training
    d2v_wrapper_fold.train_model(epochs=3)
    train_doc_vectors_labeled = d2v_wrapper_fold.infer_vocab_content_vectors()
    train_doc_vectors = train_doc_vectors_labeled.iloc[:,:-1]
    train_y = train_doc_vectors_labeled.iloc[:,-1]

    # superior classifier training
    logging.info("Fitting classifier")
    log_reg_classifier = LogisticRegression(C=0.3, solver="sag", multi_class='ovr',
                                            n_jobs=multiprocessing.cpu_count(), max_iter=1000)
    log_reg_classifier.fit(train_doc_vectors, train_y)

    # testing
    # test docs vectors inference
    test_docs_vectors = d2v_wrapper_fold.infer_content_vectors(test_docs)

    logging.info("Predicting")
    y_expected = doc_labels.iloc[test_doc_indices].values
    y_actual = log_reg_classifier.predict(test_docs_vectors)

    # evaluation:
    # logits
    logging.info("Probs collection")
    class_probs = log_reg_classifier.predict_proba(test_docs_vectors)
    class_ordered = list(log_reg_classifier.classes_)

    class_actual_index = pd.Series(y_actual).apply(lambda cat_label: class_ordered.index(cat_label))
    actual_prob = class_probs[np.arange(len(class_actual_index)), class_actual_index]

    class_expected_index = pd.Series(y_expected).apply(lambda cat_label: class_ordered.index(cat_label))
    expected_prob = class_probs[np.arange(len(class_actual_index)), class_expected_index]

    new_logits = pd.DataFrame()
    new_logits["doc_id"] = test_doc_indices
    new_logits["actual_prob"] = actual_prob
    new_logits["expected_prob"] = expected_prob
    new_logits["actual_class"] = y_actual
    new_logits["expected_class"] = y_expected
    logits = logits.append(new_logits)

    # accuracy
    logging.info("Split results:")
    split_accuracy = accuracy_score(y_expected, y_actual)
    logging.info("Run accuracy: %s" % split_accuracy)
    accuracies.append(split_accuracy)
    split_cat_accuracies = map(lambda cat: accuracy_for_category(y_expected, y_actual, cat),
                               d2v_wrapper_fold.content_categories)
    logging.info("Cat accuracies:\n%s" % split_cat_accuracies)
    cat_accuracies = cat_accuracies.append(
        pd.DataFrame(data=[split_cat_accuracies], columns=d2v_wrapper_fold.content_categories))


print "done"
print "accuracies: %s" % accuracies
print "mean accuracy: %s" % np.mean(accuracies)
print "categories accuracies: \n%s" % cat_accuracies
print "categories mean accuracies: \n%s" % cat_accuracies.mean()
print "done"
