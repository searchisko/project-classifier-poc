# System technical

## Architecture overview
The service relies on four main modules (with dependencies on the listed ones sorted bottom-up):
1. **Parsing utils**: providing the content selection logic, as well as preprocessing
of the provided resources. Further provides content de-duplication and casting to the data 
structures as expected by gensim's libraries.

2. **Doc2Vec wrapper**: In short provides a mapping of **documents to vectors**. It does so by providing 
the convenient interface to usage of Gensim's Doc2Vec implementations: 
vocabulary initialization and inference of training content that are wrapped in ``wrapper.train_model()``.
It also provides subsequent new **content vectorization** for further classification 
or possibly base for **similarity search** in document space useful e.g. for content-based recommender system. 

3. **Classifier**: interchangeable part providing the **scoring mechanism** that can be
used for a **vectorized document classification**. The default-integrated
classifier, sklearn's Logistic Regression was selected and its hyper parameters tuned based on the
analyses in 
[analytical part](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab).
The classifier can be relatively easily changed in [search_service](https://github.com/searchisko/project-classifier-poc/tree/master/search_service/search_service.py)
using ``_get_classifier_instance()``, for any other classifier providing fit(X, y) and predict_probs(X) functionality.

4. **Score tuner**: provides adaptable functionality for **customization** of possibly biased
**scoring** of documents for categories, as inferred by the selected classifier.
The module aims to **maximize** the objective **success metric** of the system so that the maximum 
performance of the **search engine** using the service is reached when **separating** relevant/irrelevant
content on **predefined score** threshold: **0.5**. More in Score Tuner section below.
