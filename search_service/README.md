# Usage

The latest deployable version of the project in a form of a web service exposes the **REST API** 
able to **score** the **relevance** of the provided piece of text towards the categories of training content.

The service is based on **Django** standalone servlet.

The data download and training process are expected to be run locally, whereas the deployable Django application
can run on remote (in our case on Openshift) with the service image created on training locally.

Before running the download and training process, make sure to set up the compatible, separated environment using
Conda. If you have not yet, [download Miniconda](https://conda.io/miniconda.html), or install it using your 
packaging system.
 
After that, from the root of the repository, run:

```commandline
# update pip to the latest version
pip install --upgrade pip
# create virtual env
conda create --name classifier python=2.7
# activate a new environment
source activate classifier
# install requirements to newly-created env
pip install -r requirements.txt
```
Within this prepared env, you can run the python scripts from the following sections.

## Content download

If you want to download a fresh content (for both RHD products, or other), proceed to 
[data section](https://github.com/searchisko/project-classifier-poc/tree/master/data).

A directory of downloaded content is then passed to training procedure (see below).

## Training

See the [training](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/training) 
section on how to create a new image of trained Service instance from the selected directory of content.

### Linking the pre-trained image

To make a deployed service to score the contents, the service needs to be pointed 
to the **image** of its **trained instance**. This can be done by either:

1. using the service class constructor, passing param ``image_dir={relative_path}`` to the constructor, 
where ``{relative_path}`` is relative to the service instance directory.

2. setting ``service.service_image_dir`` to absolute path

To make this change effective in the deployed instance,
make the according modification in Django's 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/views.py).

#### Pre-trained image

The pre-trained version of image (created 25.7.2017) is available
[here](https://drive.google.com/file/d/0B_42L5-Ve7j2STFodkprVUY0Wms/view).

This image is trained on apx. 50 000 documents, from [/data/content/prod](https://github.com/searchisko/project-classifier-poc/tree/master/data/content/prod) folder.
in addition to the relevant (products') documents, it contains a "None" category 
of 7500 non-relevant documents from StackOverflow, New York Times articles, and Political tweets.

The service image contains the models of scoring tuner, Gensim's doc2vec models, and sklearn's classifier. 
These modules can be potentially independently trained and seamlessly changed in the image directory, 
however it is not the common use case.

## Deployment

After the service is correctly linked to the trained image in 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/views.py)
and we have installed all the dependent libraries in using ``pip install -r requirements.txt``,
also in the **deployment environment**, we are ready to set up the servlet:
 
``./manage.py runserver`` or 

``sudo ./manage.py runserver``

If the server has loaded with no errors, the application is ready to respond the JSON queries.

Note the linked service image is loaded with the first request to the API, so the first request expects to take
1 - 5 seconds, depending on the hardware and model complexity.

### REST API
See the [REST Documentation](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/API).