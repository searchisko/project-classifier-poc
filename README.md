# project-classifier-poc
Proof of concept of usability of a classifier which finds the most relevant "project" for the content by it's relative content similarity with the the content of other classified documents.
The selected model should be able to categorize the unseen document by its content (and optimally other attributes).
It should also be able to provide the relevance scoring mechanism towards the categories 
so that potentionally relevant documents are to be included in search results for given categories.

The model, if successful enough, might be integrated into RH 
[content search services](https://developers.redhat.com/resources) using searchisko.

Feel free to have a look at research_notes.txt and give us a poke with any piece of a good suggestion.

Current results can be evaluated by:
1. Classification on **sentences scoring method** (provided by gensim) against standalone categories' **word2vec** models:

    cd classifiers/model/
    
    python evaluation.py

2. Classification on documents vectors as trained by **doc2vec** approach

⋅⋅⋅ a) using Scikit-learn's **Logistic Regression**

    cd classifiers/model/
    
    python d2v_superior_classifier.py

⋅⋅⋅ b) using simple **two-layered Neural Network** built on Tensorflow (more in code)

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/model/d2v_superior_classifier_neural_nb_evaled.ipynb)

The overall objectives of the project are outlined [here](https://issues.jboss.org/browse/RHDENG-1111).