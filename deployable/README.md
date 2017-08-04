# Deployable service

The latest deployable version of the project in a form of a web service exposes the **REST API** 
able to **score** the **relevance** of the provided piece of text towards the categories of training content.

The service is based on **Django** standalone servlet.

## Preparation

### Linking the pre-trained image

To make a deployed service to score the contents, the service needs to be pointed 
to the **image** of its **trained instance**. This can be done by either:

1. using the service class constructor, passing param ``image_dir={relative_path}`` to the constructor, 
where {relative_path} is relative to the service instance directory.

2. setting service.service_image_dir to absolute path

To make this change effective in the deployed instance,
make the according modification in Django's 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/deployable/search_service/views.py).

#### Pre-trained image

The pre-trained version of image (created 25.7.2017) is available
[here](https://drive.google.com/file/d/0B_42L5-Ve7j2STFodkprVUY0Wms/view).

This image is trained on apx. 50 000 documents, from [/data/content/prod](https://github.com/searchisko/project-classifier-poc/tree/master/data/content/prod) folder.
in addition to the relevant (products') documents, it containts a "None" category 
of 7500 non-relevant documents from StackOverflow, New York Times articles, and Political tweets.


The image contains the models of scoring tuner, Gensim's doc2vec models, and sklearn's classifier. 
These modules can be potentially independently trained and seamlessly changed in the image directory, 
however it is not the common use case.

See the [training](https://github.com/searchisko/project-classifier-poc/blob/master/deployable/search_service/training) 
section on how to create a new image of trained Service instance from the selected content.

## Deployment
After the service is correctly linked to the trained image in 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/deployable/search_service/views.py)
and we have installed all the dependent libraries using ``pip install -r requirements.txt``,
we are ready to set up the servlet:
 
``./manage.py runserver`` or 

``sudo ./manage.py runserver``

If the server has loaded with no errors, the application is ready to respond the JSON queries.

Note the linked service image is loaded with the first request to the API, so the first request expects to take
1 - 5 seconds, depending on the hardware and model complexity.

### REST API
The running application responds **POST** requests having JSON body in a described format on two adresses:

1a) **/score**: **request** scoring the relevance of a single document:

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

1b) **/score** **responds** in a form:

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

2a) **/scoreBulk**: **request** scoring the a bulk of documents at once. The method is much faster when scoring bigger bunch 
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

2b) **/scoreBulk** **responds** in a form:

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
