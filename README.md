# project-classifier-poc
Proof of concept of a system able to estimate the relevance of an arbitrary content towards the given categories.

The selected approach should be able to categorize the unseen document by its content (and optionally other given attributes).
In addition, it should also be able to provide the relevance scoring mechanism so that potentionally relevant content can be further included in search results for given categories.

The model, if successful, might be integrated into RH 
[content search services](https://developers.redhat.com/resources) using searchisko. It might also create a ground for more user-friendly content personalisation.

Feel free to have a look at research_notes.txt and give us a poke with any piece of a good suggestion.

Current results can be checked by:
1. Classification on **sentences scoring method** (provided by gensim) against standalone categories' **word2vec** models (currently not the best results):

    cd classifiers/model/
    
    python evaluation.py

2. Classification on documents vectors as trained by **doc2vec** approach:

⋅⋅⋅ a) (Main branch) using Scikit-learn's **Logistic Regression**

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/blob/master/classifiers/model/d2v_superior_classifier_logreg_evaled.ipynb)    

⋅⋅⋅ b) using simple **two-layered Neural Network** built on Tensorflow (more in code)

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/model/d2v_superior_classifier_neural_nb_evaled.ipynb)

Some other classifiers are tested in the same folder.

The overall objectives of the project are outlined [here](https://issues.jboss.org/browse/RHDENG-1111).