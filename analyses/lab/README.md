## Analytical

### Classifiers comparison

The deployed model was selected and evaluated based on the studies of various approaches documented below:

1. Classification on **sentences scoring method** (provided by gensim) against standalone categories' **word2vec** 
models (currently not the best results):

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/w2v_classifier_demo.ipynb)

2. Classification on documents vectors as trained by **doc2vec** approach:

⋅⋅⋅ a) (Main branch) using **Logistic Regression**, hyper-tuning the C param

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/blob/master/analyses/lab/d2v_superior_classifier_logreg_evaled.ipynb)    

... Including also doc2vec training to Cross-Validation (seems to make no big difference): 
[notebook](https://github.com/searchisko/project-classifier-poc/blob/master/analyses/lab/d2v_superior_classifier_logreg_evaled_deepeval.ipynb)

⋅⋅⋅ b) using simple **two-layered Neural Network** built on Tensorflow (more in code)

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/d2v_superior_classifier_neural_nb_evaled.ipynb)

... c) using **Support Vector Machines** with linear and radial kernels (performed the best)

... See [linear kernel notebook](https://github.com/searchisko/project-classifier-poc/blob/master/analyses/lab/d2v_superior_classifier_lin_svm_evaled.ipynb)
or [radial kernel notebook](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/d2v_superior_classifier_svm_evaled.ipynb)

The approach 2a) has been for it's simplicity and a feature of native scoring method 
(probabilities prediction of Logistic Regression) selected to be followed up and implemented 
into the [deployable service](https://github.com/searchisko/project-classifier-poc/blob/master/deployable/search_service).

### Scoring analyses
* See the visualization of transformation of the scores for a selected category by ScoreTuner module
[here](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/cat_score_viewer.ipynb)

* See the overall evaluation of the. final scoring of the Search Service 
on both relevant and irrelevant data sets:

1. Of the service **not including** the **"None"** category in training content: 
[here](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/score_tuning_analysis_standalone-none_excl.ipynb)

2. Of the service **including** the **"None"** category in training content: 
[here](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/score_tuning_analysis_standalone-none_incl.ipynb)

These contains the statistical demonstration of scoring of both relevant content, 
as well as the selected distinct irrelevant data sets - Political Tweets, 
Economical Articles from NY Times, StackOverflow questions.

The irrelevant data sets are distinct from the ones that the service was trained on.

The relevant data set was scored in CV manner, meaning that the classifier was trained
on a distinct data set. Vector inference module was however trained on whole content
at once, this has however proved to very little affect the classification performance, 
as demonstrated in comparison of classifier's 
[standard evaluation](https://github.com/searchisko/project-classifier-poc/blob/master/analyses/lab/d2v_superior_classifier_logreg_evaled.ipynb)
and [deep evaluation](https://github.com/searchisko/project-classifier-poc/blob/master/analyses/lab/d2v_superior_classifier_logreg_evaled_deepeval.ipynb)
referred as in **Classifier comparison: 2a)**.