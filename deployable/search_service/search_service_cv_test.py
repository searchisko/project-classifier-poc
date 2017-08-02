from search_service import RelevanceSearchService
from dependencies.doc2vec_wrapper import D2VWrapper
from dependencies import parsing_utils as parsing

from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np

import logging
from sklearn.externals import joblib

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

base_d2v_wrapper = D2VWrapper()

content_basepath="../../data/content/prod_sample"
content_categories = ["amq", "eap", "webserver", "datagrid", "fuse", "brms", "bpmsuite", "devstudio", "cdk",
                "developertoolset", "rhel", "softwarecollections", "mobileplatform", "openshift"]
# content_categories = ["amq", "webserver", "datagrid"]


all_content_df = parsing.get_content_as_dataframe(content_basepath, "_content.csv", content_categories).drop_duplicates()
# doc_content = parsing.select_training_content(all_content_df).apply(lambda word_list: parsing.content_from_words(word_list))
doc_content = all_content_df["sys_content_plaintext"]
doc_headers = parsing.select_headers(all_content_df).apply(lambda content: "" if np.any(pd.isnull(content)) else content) \
    .apply(lambda word_list: parsing.content_from_words(word_list))

doc_content.index = doc_headers.index
doc_ids = pd.Series(range(len(doc_headers)), index=doc_content.index)
y = all_content_df["target"]

docs_df = pd.DataFrame(columns=["content", "headers"], index=doc_content.index)
docs_df["content"] = doc_content
docs_df["headers"] = doc_headers
docs_df["y"] = y

# drop duplicates from the set
docs_df = docs_df[~docs_df.duplicated(subset=["headers", "content"])]

splits = 5
strat_kfold = StratifiedKFold(n_splits=splits, shuffle=True)
logging.info("Gathering training content scores in %s splits" % splits)

docs_scores = pd.DataFrame(columns=content_categories+["y"])

for train_doc_indices, test_doc_indices in strat_kfold.split(docs_df, docs_df["y"]):
    service = RelevanceSearchService()

    train_docs_df = docs_df.iloc[train_doc_indices]
    test_docs_df = docs_df.iloc[test_doc_indices]

    service.train_on_docs(doc_ids=train_docs_df.index, doc_contents=train_docs_df["content"],
                          doc_headers=train_docs_df["headers"], y=train_docs_df["y"])

    test_docs_scores, _ = service.score_docs_bulk(doc_ids=test_docs_df.index, doc_contents=test_docs_df["content"],
                                               doc_headers=test_docs_df["headers"])
    test_docs_scores["y"] = test_docs_df["y"]

    docs_scores = docs_scores.append(test_docs_scores)

print docs_scores.describe()

scores_pickle_path = "scores_pickled_wout_none.dump"
logging.info("Pickling scores to %s" % scores_pickle_path)
joblib.dump(docs_scores, scores_pickle_path)

print "done"
