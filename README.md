# Content relevance scorer for RH products

A system able to estimate the relevance of an arbitrary content towards the learned categories.

The system is able to score the unseen document by its content 
(and potentially other attributes) based on its similarity to the seen ones.
It also provides the matching scoring mechanism enabling the potentially relevant content 
to be included in search results for trained categories.

The system will be integrated into RH 
[content search services](https://developers.redhat.com/resources) using DCP. 

It might also be further extended to provide a smart content-based recommender system for web portals 
with sufficient amount of training documents (regardless of categorization).

The project contains two main components:

1. [Deployable service](https://github.com/searchisko/project-classifier-poc/tree/master/deployable) 
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

2. [Content downloader](https://github.com/searchisko/project-classifier-poc/tree/master/data)
providing tools for convenient bulk download of the indexed content (of **DCP** and **access.redhat**)
categorized to the Red Hat products.

In addition to that, the project contains the 
[analytical part](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab) 
that has driven the best selection of the classifiers and configuration of the system parameters.

The overview and the rough evaluation of the system are presented in both 
[technical](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/slides/overview_presentation_nlp.pdf)
and 
[promotional](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/slides/ML_for_RHD.pdf)
presentations.

If you're mostly interested in technical part of the project, also reach out for the 
[technical documentation](https://github.com/searchisko/project-classifier-poc/tree/master/deployable/technical_docs)
of the service.

Various further evaluation of the current system by some more tricky metrics are summed up in the most
[current analyses](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/score_tuning_analysis_standalone-none_incl.ipynb).

The overall progress and objectives are tracked [here](https://issues.jboss.org/browse/RHDENG-1111).