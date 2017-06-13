from classifiers.model.service.search_service import RelevanceSearchService


service = RelevanceSearchService()

# EITHER perform a correct train and persist phase
service.train(train_content_dir="../../../data/content/prod")
service.evaluate_performance()
service.persist_trained_model()

# OR load a persisted model and automatically estimate its performance respectively to the combined-F measure
# introduced in score tuner
# service.load_trained_model()
# service.evaluate_performance()

print "done"
