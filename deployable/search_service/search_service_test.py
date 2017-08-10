import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from search_service import ScoringService
trained_service = ScoringService()

# trained_service.train("../../data/content/no_preproc")
# trained_service.persist_trained_model(persist_dir="service_no_preproc")

# if training with new config, or for the first time, evaluate the service performance
# could be VERY time-consuming
trained_service.evaluate_performance(eval_content_dir="../../data/content/no_preproc")

# create an image of the trained service

# EITHER perform a correct train and persist phase
# service.train(train_content_dir="../../data/content/prod_sample")
# service.train(train_content_dir="data/content/prod_sample")
# service.persist_trained_model()
# service.evaluate_performance()

# OR load a persisted model and automatically estimate its performance respectively to the combined-F measure
# introduced in score tuner
# print trained_service.score_doc(1, "jboss", "jboss")

# service.evaluate_performance()

print "done"
