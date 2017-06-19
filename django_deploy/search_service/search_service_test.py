from search_service import RelevanceSearchService


service = RelevanceSearchService()

# EITHER perform a correct train and persist phase
service.train(train_content_dir="../../data/content/prod_sample")
# service.train(train_content_dir="data/content/prod_sample")
service.persist_trained_model()
# service.evaluate_performance()

# OR load a persisted model and automatically estimate its performance respectively to the combined-F measure
# introduced in score tuner
# print service.score_doc(1, "jboss", "jboss")

# service.load_trained_model()
# service.evaluate_performance()

print "done"
