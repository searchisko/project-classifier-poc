# Relevance scorer for RH products

A system able to estimate the **relevance of an arbitrary content** towards the learned categories.

The system is able to **score** the unseen document by its content 
(and potentially other attributes) based on its **contextual similarity** to the seen ones.
It also contains the **score tuning** mechanism enabling the direct use of the documents' relevance scores 
by a **search engine** filtering the relevant/irrelevant results by a single fixed threshold and easily
reaching the optimal performance.

The system will be integrated into 
[RH content search services](https://developers.redhat.com/resources) using DCP content indexing tool. 

It might also be further extended to provide a smart **content-based recommender system** for web portals 
with sufficient amount of training documents (regardless of categorization).

The project currently contains two **main components**:

1. **[Deployable search service](https://github.com/searchisko/project-classifier-poc/tree/master/search_service)**
providing intuitive REST API for scoring an arbitrary content towards the trained categories:

Request:
```json
{
	"sys_meta": false,
	"doc": {
		"id": "DOC_123",
		"title": "One smart doc",
		"content": "This is one dull piece of text."
	}
}
```

Response:
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

2. **[Content downloader](https://github.com/searchisko/project-classifier-poc/tree/master/data)**
providing tools for convenient bulk download of the indexed content (of **DCP** and **access.redhat**)
categorized towards the Red Hat products.

In addition to that, the project contains the 
[analytical part](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab) 
that has driven the selection of the classifier and configuration of the system parameters.

The architecture and the technologies used are briefly introduced in
[overview presentation](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/slides/ML_for_RHD.pdf)
and slightly 
[technical presentation](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/slides/overview_presentation_nlp.pdf).

If you're interested in **technical background of the project**, try to understand the 
**[technical documentation](https://github.com/searchisko/project-classifier-poc/tree/master/search_service/technical_docs)**
of the system.

Various further evaluation of the current system by some more tricky metrics are summed up in the most
[fresh analysis](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/score_tuning_analysis_standalone-none_incl.ipynb).

The overall progress and objectives of the project are tracked [here](https://issues.jboss.org/browse/RHDENG-1111).

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

The pre-trained version of image (created 14.6.2018) is available 
[here](https://drive.google.com/file/d/0B_42L5-Ve7j2STFodkprVUY0Wms/view).

See a readme of [pretrained images](https://github.com/searchisko/project-classifier-poc/tree/master/search_service/pretrained_service_images)
for a description and other pretrained images of older gensim versions.
