from classifiers.model.service.search_service import RelevanceSearchService

import pandas as pd

books_csv_df = pd.read_csv("../../../data/content/books_test/books_no_preproc.csv",
                           dtype={'sys_content_plaintext': str})

books_ids = pd.Series(range(0, len(books_csv_df)))
books_headers = books_csv_df["sys_title"]
books_content = books_csv_df["sys_description"]

service = RelevanceSearchService()

# service.load_trained_model()
# service.evaluate_performance()

one_doc_scores = service.score_doc(books_ids.iloc[9], books_headers.iloc[9], books_content.iloc[9])

all_docs_scores = service.score_docs_bulk(books_ids.iloc[7:8], books_headers.iloc[7:8], books_content.iloc[7:8])

print "done"
