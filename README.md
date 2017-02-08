# project-classifier-poc
POC of classifier which finds the most relevant "project" for the content by it's relative semantic similarity with the classified content.

Currently in a state of preprocessing and parameters tuning of Word2Vec models. Other models will be considered as well as soon as the W2V model will be correctly set and evaluated.

Feel free to have a look at research_notes.txt and discuss the further directions of the effort.

Current results can be evaluated by:

`cd word2vec_classificator/`

`python evaluation.py`

The overall objectives of the project are outlined [here](https://issues.jboss.org/browse/RHDENG-1111).