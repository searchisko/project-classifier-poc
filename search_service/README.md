# Usage

The latest deployable version of the project in a form of a web service exposes the **REST API** 
able to **score** the **relevance** of the provided piece of text towards the categories of training content.

The service is based on **Django** standalone servlet.

## Preparation

### Linking the pre-trained image

To make a deployed service to score the contents, the service needs to be pointed 
to the **image** of its **trained instance**. This can be done by either:

1. using the service class constructor, passing param ``image_dir={relative_path}`` to the constructor, 
where ``{relative_path}`` is relative to the service instance directory.

2. setting ``service.service_image_dir`` to absolute path

To make this change effective in the deployed instance,
make the according modification in Django's 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/API/views.py).

#### Pre-trained image

The pre-trained version of image (created 25.7.2017) is available
[here](https://drive.google.com/file/d/0B_42L5-Ve7j2STFodkprVUY0Wms/view).

This image is trained on apx. 50 000 documents, from [/data/content/prod](https://github.com/searchisko/project-classifier-poc/tree/master/data/content/prod) folder.
in addition to the relevant (products') documents, it containts a "None" category 
of 7500 non-relevant documents from StackOverflow, New York Times articles, and Political tweets.


The image contains the models of scoring tuner, Gensim's doc2vec models, and sklearn's classifier. 
These modules can be potentially independently trained and seamlessly changed in the image directory, 
however it is not the common use case.

See the [training](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/API/training) 
section on how to create a new image of trained Service instance from the selected content.

## Deployment
After the service is correctly linked to the trained image in 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/API/views.py)
and we have installed all the dependent libraries using ``pip install -r requirements.txt``,
we are ready to set up the servlet:
 
``./manage.py runserver`` or 

``sudo ./manage.py runserver``

If the server has loaded with no errors, the application is ready to respond the JSON queries.

Note the linked service image is loaded with the first request to the API, so the first request expects to take
1 - 5 seconds, depending on the hardware and model complexity.

### REST API
See the [REST Documentation](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/API).