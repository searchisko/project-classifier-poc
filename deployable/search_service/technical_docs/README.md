## System technical

### Architecture overview
The service relies on four main modules (dependency hierarchy bottom-up):
1. **Parsing utils**: providing the selection logic of the content including preprocessing
of the provided resources. Further provides content de-duplication, casting to the data 
structures as expected by gensim's libraries.

2. **Doc2Vec wrapper**: Provides a mapping of **documents to vectors**. In addition provides 
the convenient interface to usage of Gensim's Doc2Vec implementations: 
vocab initialization, inference model training and subsequent 
new **content vectorization** for further classification or possibly base for **similarity search** 
in document space. 

3. **Classifier**: interchangeable part providing the base **scoring mechanism** that can be
used for an arbitrary **vectorized document classification**. The used
classifier, sklearn's Logistic Regression was selected and its hyper parameters tuned based on the
analyses in 
[analytical part](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab).
The classifier can be relatively easily changed in [search_service](https://github.com/searchisko/project-classifier-poc/tree/master/deployable/search_service/search_service.py)
using ``_get_classifier_instance()``, for any other classifier providing fit(X, y) and predict_probs(X) functionality.

4. **Score tuner**: provides adaptable functionality for customization of possibly biased
scores of documents for categories, as inferred by the selected classifier.
The module aims to **maximize** the objective **success metric** of the system so that the maximum 
performance of the **search engine** using the service is reached when **separating **relevant/irrelevant
content on **predefined score** threshold: **0.5**. More in training process explained.

### Training process explained
Providing the directory of the training content as an argument of ``service.train(directory)``
will trigger the scan of the directory, expecting the files to contain the distinct categories,
with categories names as prefixes of these files. After the list of resources is known,
the resources are loaded and training content is parsed to ``CategorizedDocuments``.

Note that by default the training process in expected to be preprocessed using
``text_preprocess.preprocess_text(text)``. In other case, preprocessing method **must**
be matched on scoring phase, by passing the preprocessing procedure to
``score_docs(preprocess_method=method)`` or ``score_docs_bulk(preprocess_method=method)``.

The ``CategorizedDocuments`` are in Doc2Vec Wrapper used for building word vocabulary 
and subsequent training documents vector inference. The base doc2vec model is cyclically trained 
on the shuffled content in multiple epochs. This is documented and has proved to slightly improve the
following classification performance., thus supposedly improves the doc2vec model accuracy.

On the service layer, the provided classifier is then fitted using ``classifier.fit(doc_vectors)``. This classifier is later 
used for classification of the new (vectorized) documents. The selection of the default classifier
was the result of comparative analysis based on the accuracy of classification by the classifiers.
The comparison of classifiers performance is compared on 
[slides here](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/slidesoverview_presentation_nlp.pdf).

Since it's been observed in 
[classifiers performance analyses](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab) 
that the classifiers tend to weight the categories scoring unequally, 
the service gather the scores of training content analyzes it using 
[Score Tuner]((https://github.com/searchisko/project-classifier-poc/tree/master/deployable/search_service/score_tuner.py)) 
module (below) and acts upon the identified bias using Score Tuner's scores tuning.

The training documents scores are obtained in **Cross-Validation** manner: the classifier 
with selected configuration is always **trained** on e.g. **4/5** of of the content, 
able to adequately **score** the remaining **1/5**. Repeating the process, the service gathers
the scores of all training content to train the Score tuner on it 
([service]((https://github.com/searchisko/project-classifier-poc/tree/master/deployable/search_service/score_tuner.py))
's ``_score_train_content()`` method)

#### Score Tuner
[Score Tuner](https://github.com/searchisko/project-classifier-poc/tree/master/deployable/search_service/score_tuner.py)
estimates the threshold for every category by maximizing
the chosen precision/recall (F-score's beta) ratio on so called ``general_search_threshold``
(``=0.5`` by default).
This is performed easily by solving the scalar **function minimization** problem where the
minimized objective function is the inverse of F-score with selected beta. 
(method ``maximize_f_score()``)

This procedure will for **each category** produce the **optimal separation threshold** that separates
the documents (by its scores) so that the maximum desired performance is reached. Therefore,
to reach this performance on ``general_search_threshold`` we linearly transform the scores
of each category so that the scores of **optimal separation threshold** scale to 
``general_search_threshold`` (method ``transform_probs(_df)``).

Thus after learning the **``cats_original_thresholds``**, we can perform this transformation
as well on a new content, expecting that the performance of the search engine should keep
unbiased of the selected classifier tendencies.

The alchemy still remains in an appropriate selection of the **betas** for categories' maximized
F-scores. This have a crucial effect on a selection of the categories optimal **separation 
thresholds**. However this obviously remains a matter of user (search engine) **preferences**:
whether to retrieve more content in a price of giving more false positives, 
or to retrieve less in a price of leaving off some important results.

We have decided to balance this problem based on **categories size**, so that the categories
with large data set will gain only the content that is almost certainly relevant, and small categories
might get the content that is rather marginally relevant. 
For that we used log-linear scaling of betas inversely proportional to the categories' size.
This in consequence significantly outweights the precision in case of eap, fuse or openshift,
whereas outweights recall on small categories.

Whether this is good or bad might suggest perhaps only the analyses of bahavior 
on independent irrelevant data set. See comparison of None category included and excluded 
in training in [lab section](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab)  
for that.

##### Experimental: ``tune_betas_by_none()``

There is a heuristic for a fine tuning of the betas that threats the beta bound as free parameters subjecting
the minimization process of the objective function estimating the performance of the tuner.

The performance function **harmonically** balances the performance of the system on positive and negative
categories content, minimizing its value when **minimizing the number of the irrelevant** data set with
scores of any category above 0.5 and **maximizing the number of correctly classified** docs of the relevant
 (=categorized) data set.
 
This function is however still just a heuristic where the weighting of the two sides (relevant-irrelevant)
is a choice of the user. Furthermore, the function has proven to retrieve drastically various results for
close inputs, suggesting that it has a great number of local minimums.

For this reason, it is suggested to **select** the **betas borders** by onw **expert knowledge**, and not to
use the function for more than just a fine tuning of the pre-thought borders.

#### Training metadata

After the training is finished, the service persists the statistics identifying the training content 
and providing additional statistics of training. Once the service is evaluated using 
``evaluate_performance()``, the results of the evaluation are also available. These metadata are
held in ``service_meta`` class variable.

### New content scoring process
Once all the **Doc2Vec**'s base model, vector **classifier** and **Score Tuner** are trained,
the system is able to adequately score the new provided content. For consistency 
with training process, the service only provides the scoring of the content 
having both title and body content.

The scoring process follows the steps of training process, only performing
the predictions instead of modules training.

First, the provided texts are **preprocessed** using the providing preprocessing method. 
The **preprocessing** method applied to training content **should match** the preprocessing passed to
``service.score()`` or ``service.score_bulk()`` methods.

The preprocessed docs are then **vectorized** using the pre-embedded Doc2Vec model of **Doc2VecWrapper**.
The vector inference based on the trained model has been proved to be a source of a considerable entropy.
After the Score Tuner multiplication of lower-scored category, the **scores** for given doc and given category
might **bounce** for as much as 50%.

To stabilize the inference process, the Doc2Vec wrapper performs the averaging of the vectors inferred
in the selected number of cycles (using ``d2v_wrapper.infer_content_vectors(infer_cycles=10)`` method param.)

After the vector inference, the passed doc(s) are **scored** using the selected classifier, providing the
``predict_probs(X)`` method.

The inferred probs are then **transformed** according to the Score Tuner's optimal
thresholds for the classified categories during the training phase.

The transformed probabilities are returned as relevance scores for categories of the given documents.

### Evaluation

To perform the proper evaluation of the service, the ``evaluate_performance()``
method does not use the pre-trained instance of the Service, but rather trains and test the new
instance of the Service with the configuration of the current instance. This is to **ensure
the distinctivity** of the training and testing data set as well as the respective ratio of categories' size.

The evaluation procedure splits the train/test content in Cross Validation manner, and in every split
trains the new instance of the Service on the bigger part. The scoring is performed using the
``score_bulk()`` method of the service so the test covers everything below. 

The **positive performance metric** used is rather non-intuitive **combination of f-scores** 
of classified categories, with the betas as defined by ``score_tuner.target_beta_scaling`` range
and with true/false separate threshold = 0.5

The **negative performance metric** is a **ratio** of **irrelevant content falsely classified** as 
relevant for some of the relevant categories by true/false separate threshold = 0.5.

See the logs of the evaluation for more detailed analysis of the performance of the Service.

The services with various configurations can be compared using the listed metrics, however the precise
**weighting** of these metrics remain dependent to an **expert knowledge** of the user and should be precisely 
considered.

**Note** that the evaluation performs ``<folds>`` of trainings of the whole Service, where evaluation of
one split can take around 3hrs. Avoid running the procedure repeatedly, if the configuration did not 
change significantly.

### Modules free parameters
A list of the configurable parameters of the system and a brief description of its impact to the functionality.

#### Doc2Vec wrapper
* **vector_length**: documents' inferred vectors length. Increment of default 300 to 800+800 (doc header+content)
has proved to increase the classification accuracy. Other classifiers has proved to perform better 
with different values, e.g. described neural network as a classifier performed better with 500+500.
* **window length**: length of the sliding window (in words) that the word2vec uses for 
computing the context probabilities. Default (=8) has proved to work the best for classification.
* **train_algo**: dbow (distributed bag of words) or dm (distributed memory). Dbow has proved to work faster
as well as slightly more accurate, thus has been selected default
* **drop_short_docs**: marginal document length (in words) under which the documents are dropped from training.
Defaults to 10. Increases classification accuracy by ~1%.
* **train epochs**: IN how many cycles the training content is shuffled and sequentially fed to the 
training doc2vec algorithm. Defaults to 10, does improve accuracy marginally after cca. 5.
* **vector inference**: infer_alpha, infer_subsample, infer_steps: vector inference parameters 
determining smoothing, subsampling and inference precision respectively. Changing the default has proved to
change the classification accuracy marginally.
* **inference cycles**: Number of averaging cycles for vectors inference. The resulting document vector
is an average of ``<inference cycles>`` inferences.

#### Classifier
Dependent on a type of classifier: in case of Logistic regression:
* **C**: regularization power, determines how much the errors are significant
* **multiclass algorithm**: approach to solving the discrimination of multi class classification. Defaults
to one-versus-rest (ovr).

The parameters of the classifier are and should be appropriately hyper-tuned regarding the objective metric.

The C in case of used Logistic Regression classifier optimized to 0.3 observing Accuracy metric,
in compare to 0.22 when observing the introduced weighted F-scores for categories with customized betas metric.


#### Score Tuner
Needs to have either trained ``cats_original_thresholds``, or train them with respect to
the set of ``target_beta_scaling``.

* **beta range**: F-scores betas (or harmonic precision/recall ratios) desired to be maximimized
independently for each scored category when searching for optimal ``cats_original_thresholds``
* **optimal separate thresholds**: separate thresholds for categories can as well be directly set,
without minimizing the beta range.
* **general_search_threshold**: threshold that is expected to be used by the search engine, to distinguish
the relevant from irrelevant content. The Score Tuner learns to transform the scores so that the search
engine eventually performs the best when dividing the content by transformed scores on the selected 
``general_search_threshold``. Defaults to **0.5**.

#### Service training
* **training content scoring splits**: Number of the splits used for inference of the scores for training content.
In general, the higher the number of splits, the more accurate the training of the ScoreTuner, since the
classifier is in more splits trained on bigger data set. Practically, however, the change in result is marginal
and cyclical training of scoring classifier might be timely consuming (appx. 30min for one 5-fold split).

### System performance
The **training** is a long-running task where the rough estimates of the time lapses are as followed:
* Appx 15min on one epoch of training on 50k documents. Default 10 epochos might take appx. 2.5hrs.
* Appx 20min on train content vector inference. This is repeated ``<inference cycles>`` times.
* Appx 20min on training each split classifier for training documents' scoring. This is repeated 
``<training content scoring splits>`` times.
* Appx 20min on training the main classifier on all training vectors. This is one time job.
* Appx 30secs to train the Score Tuner.
* Appx 1min to persist the trained model

The system **evaluation** performs the task of training the new Service ``<splits>`` times cyclically in CV with
the same amount of train/test data, so the overall time of evaluation is ``<splits>`` times `train()`

The **scoring** of a **new content** takes:
* Appx 5secs for the first time, when the persisted model is reloaded together with scoring, depending on the
disk performance
* Appx 0.1secs for every other query of one document: ``service.score_doc()``
* Appx 1minute for ``service.score_docs_bulk()`` with 10k documents
* TODO ^^ customize or extend after detailed performance testing
