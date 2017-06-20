# Project classifier for RH sites content
Proof of concept of a system able to estimate the relevance of an arbitrary content towards the given categories.

The selected approach should be able to categorize the unseen document by its content (and optionally other given attributes).
In addition, it should also be able to provide the relevance scoring mechanism so that potentially relevant content can be further included in search results for given categories.

The model, if successful, might be integrated into RH 
[content search services](https://developers.redhat.com/resources) using searchisko. It might also create a ground for more user-friendly content personalisation.

Feel free to have a look at research_notes.txt and give us a poke with any piece of a good suggestion.

## Deployable service

The latest deployable version of the research in a form of a web service exposes the REST API able to score the relevance of the provided content towards the modeled categories.

The service is based on Django standalone container.

To test out, clone the repo (containing the pre-trained models) to CLONE_DIR. Then:

```bash
export PYTHONPATH=<CLONE_DIR>/project-classifier-poc/django_deploy/search_service
cd project-classifier-poc
pip install -r requirements.txt
cd django_deploy
./manage.py runserver
```

The server should by default respond **GET** and **POST** requests containing params: **doc_id, doc_header, doc_content** on: **http://localhost:8000/score**

## Development

Current results can be checked by:
1. Classification on **sentences scoring method** (provided by gensim) against standalone categories' **word2vec** models (currently not the best results):

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/model/w2v_classifier_demo.ipynb)

2. Classification on documents vectors as trained by **doc2vec** approach:

⋅⋅⋅ a) (Main branch) using Scikit-learn's **Logistic Regression**

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/blob/master/classifiers/model/d2v_superior_classifier_logreg_evaled.ipynb)    

⋅⋅⋅ b) using simple **two-layered Neural Network** built on Tensorflow (more in code)

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/model/d2v_superior_classifier_neural_nb_evaled.ipynb)

Some other classifiers are tested in the directory in the same manner.


The overall objectives of the project are outlined [here](https://issues.jboss.org/browse/RHDENG-1111)

and in slides [here](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/info).
