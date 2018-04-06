## Training the service
The service can be conveniently trained on an arbitrary set of categories' documents that are collected and organized as described below.

In case of scoring the relevance of RH products, all that needs to be done to organize the content in desired format is running a 
[products_downloader.py](https://github.com/searchisko/project-classifier-poc/tree/master/data/products_downloader.py)
that will download the content indexed in DCP and access.redhat.com, that is somehow mapped to the RH products.

If you want to train on the fresh content, take a look at 
[data section](https://github.com/searchisko/project-classifier-poc/tree/master/data) 
for how to gather it.

After the preprocessed content is successfully downloaded to the given directory,
the search service can be trained on it, and the image of the trained service can be created
to be then used for scoring the new content.

To train the new instance of the search service, evaluate it and export the trained image, do:

```python
from search_service import ScoringService
trained_service = ScoringService()
trained_service.train(train_content_dir="downloaded/content/dir")

# if training with new config, or for the first time, evaluate the service performance
# could be VERY time-consuming
trained_service.evaluate_performance(eval_content_dir="downloaded/content/dir")

# create an image of the trained service
trained_service.persist_trained_model(persist_dir="training/new_image_dir")
```

You can set the preprocessing method used for train and scored text through the Service classifier:
```python
trained_service = ScoringService(preprocessing=my_preproc_method)
```
by default, the native ``preprocess_text(text, stemming=False)`` method from 
[text_preprocess](https://github.com/searchisko/project-classifier-poc/tree/master/search_service/API/dependencies/text_preprocess.py)
will be used.

Note that the train and especially evaluation routine might take several hours on tens of thousands of documents.
See the Service's [technical docs](https://github.com/searchisko/project-classifier-poc/tree/master/search_service/API/technical_docs) 
for the explained description of the training process.

### Training on other data sources
The system can be trained on own set of documents in a specified format, and of arbitrary categories.

The **``train(directory)``** routine of the service expects as its argument 
a **folder** containing the data independently stored in a **separate file** for each category.
The naming convention of the file is also mandatory to oblige: each data file has a name in a form of:
**\<category\><_suffix>.csv**, e.g. **\"eap_content.csv\"**.

The training csv files must contain the **documents as rows** with the following attributes (colunms):
* **sys_title**: Document short header, if available. Used for training if **sys_description** is empty.
* **sys_description**: Document description, expected to be longer than **sys_title**. 
If empty, **sys_title** is used for training instead.
* **sys_content_plaintext**: Main piece of text of the document. Always used for training. Mandatory.
* **source**: Document source, e.g. stackoverflow. Can be left empty.
* **target**: Target category, matching the prefix of a file of this document.

The training csv-s, as well as the scored contents are expected **not preprocessed**. 
The preprocessing using the given method is performed before the training on training content and
before the scoring on scored content.

**If planning to download the training content**, you can take a look into some implemented 
[data retrieval procedure](https://github.com/searchisko/project-classifier-poc/tree/master/data/searchisko_requestor.py)
to see how to perform a download into the predefined format.

**If parsing the resources from csv-s**, you can follow the 
[dataset preparation](https://github.com/searchisko/project-classifier-poc/tree/master/analyses/lab/dataset_preparation.ipynb) 
framework used for parsing the irrelevant data sets.
