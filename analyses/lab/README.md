## Analytical

The deployed model was selected and evaluated based on the studies of various approaches documented below:

1. Classification on **sentences scoring method** (provided by gensim) against standalone categories' **word2vec** 
models (currently not the best results):

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/w2v_classifier_demo.ipynb)

2. Classification on documents vectors as trained by **doc2vec** approach:

⋅⋅⋅ a) (Main branch) using **Logistic Regression**, hyper-tuning the C param

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/blob/master/analyses/lab/d2v_superior_classifier_logreg_evaled.ipynb)    

⋅⋅⋅ b) using simple **two-layered Neural Network** built on Tensorflow (more in code)

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/d2v_superior_classifier_svm_evaled.ipynb)

... c) using **Support Vector Machines** with linear and radial kernels (performed the best)

Some other classifiers are tested in the directory in the same manner.

The approach 2a) has been for it's simplicity and a feature of native scoring method 
(class probabilities inference of Logistic Regression) selected to be followed up and implemented 
into the [deployable service](https://github.com/searchisko/project-classifier-poc/blob/master/deployable/search_service)
