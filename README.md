# Project classifier for RH sites content
A service able to estimate the relevance of an arbitrary content towards the learned categories.

The system is able to score the unseen document by its content 
(and potentially other attributes) based on its similarity to the seen ones.
It is also able to provide the relevance scoring mechanism enabling the potentially relevant content 
to be included in search results for trained categories.

The system might be integrated into RH 
[content search services](https://developers.redhat.com/resources) using Searchisko. 

It might also be further extended to provide a smart content-based recommender system for web portals 
with sufficient amount of training documents (regardless of categorization).

Feel free to give us a poke with any piece of an idea about stuff.

## Deployable service

The latest deployable version of the research in a form of a web service exposes the REST API able to score 
the relevance of the provided content towards the modeled categories.

The service is based on Django standalone container.

To test out, clone the repo (containing the pre-trained models) and then:

```bash
cd project-classifier-poc
pip install -r requirements.txt
cd django_deploy
./manage.py runserver
```

### REST API
The running application responds **POST** requests having JSON body in a described format on two adresses:

1. **/score**: scoring the relevance of a single document:

```json
{
	"sys_meta": false,
	"doc": {
		"id": "DOC_123",
		"title": "One smart doc",
		"content": "This document has a lot of funny stuff."
	}
}
```

**/score** responds in a form:

```json
{
    "scoring": {
        "softwarecollections": 0.000060932611962771777,
        "brms": 0.00080337037910394038,
        "bpmsuite": 0.00026477703963384558,
        "...": "..."
    }
}
```

1. **/scoreBulk**: scoring the a bulk of documents at once. The method is much faster when scoring bigger bunch 
of docs at once, than :

```json
{
	"sys_meta": true,
	"docs": {
		"DOC_123": {
		    "title": "One smart doc",
		    "content": "This document has a lot of funny stuff."
		},
		"DOC_234": {
		    "title": "One silly doc",
		    "content": "This doc is not as funny as the first one."
		}
	}
}

```

**/scoreBulk** responds in a form:

```json
{
    "scoring": {
        "DOC_123": {
            "softwarecollections": 0.000064274845465339681,
            "brms": 0.0009433698353256692,
            "...": "..."
        },
        "DOC_234": {
            "softwarecollections": 0.000032687001843056951,
            "brms": 0.002657126406253485,
            "...": "..."
        }
    },
    "sys_meta": {
        "response_status": "OK",
        "request_time": "2017-06-22T13:38:07.666230",
        "response_time": "2017-06-22T13:38:07.723830",
        "sys_model_training_time": "2017-06-15T12:55:01.981282"
    }
}
```

**Note** that **sys_meta** object is included in a response if ```"sys_meta": true``` is set in request. 
It defaults to ```false```.

### Training on own data
The system can be trained on own set of documents in a specified format. 
TODO: document the format of input data. The current model was trained on apx. 50 000 documents
of [/data directory](https://github.com/searchisko/project-classifier-poc/tree/master/data/content/prod).

## Development

The deployed model was selected and evaluated based on the studies of various approaches documented below. 
Check them out if you're interested to see how smart the guy is.

1. Classification on **sentences scoring method** (provided by gensim) against standalone categories' **word2vec** 
models (currently not the best results):

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/model/w2v_classifier_demo.ipynb)

2. Classification on documents vectors as trained by **doc2vec** approach:

⋅⋅⋅ a) (Main branch) using Scikit-learn's **Logistic Regression**

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/blob/master/classifiers/model/d2v_superior_classifier_logreg_evaled.ipynb)    

⋅⋅⋅ b) using simple **two-layered Neural Network** built on Tensorflow (more in code)

⋅⋅⋅ See [demo notebook](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/model/d2v_superior_classifier_neural_nb_evaled.ipynb)

Some other classifiers are tested in the directory in the same manner.

The approach 2a) is packed in a deployable service.

The overall objectives of the project are outlined [here](https://issues.jboss.org/browse/RHDENG-1111)

and in slides [here](https://github.com/searchisko/project-classifier-poc/tree/master/classifiers/info).
