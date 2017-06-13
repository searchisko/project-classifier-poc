from classifiers.model.service.search_service import RelevanceSearchService


service = RelevanceSearchService()
service.train(train_content_dir="../../../data/content/prod")
service.evaluate_performance()
service.persist_trained_model()
# service.load_trained_model()
# service.evaluate_performance()
print "done"
