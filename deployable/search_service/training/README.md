## Training the service
The service can be conveniently trained on an arbitrary set of categories' documents that are collected and organized as described below.

In case of scoring the relevance of RH products, all that needs to be done to organize the content in desired format is running a 
[downloader_automata.py](https://github.com/searchisko/project-classifier-poc/tree/master/data/downloader_automata.py)
that will download the content indexed in DCP and access.redhat.com, that is somehow mapped to the RH products.

If you want to train on the fresh content, take a look at 
[data section](https://github.com/searchisko/project-classifier-poc/tree/master/data) 
for how to gather it.

After the preprocessed content is successfully downloaded to the given directory,
the search service can be trained on it, and the image of the trained image can be created
to be then used for scoring the new content.

To train the new instance of the search service, evaluate it and export the trained image, do:

```python
from search_service import RelevanceSearchService
trained_service = RelevanceSearchService()
trained_service.train(train_content_dir="downloaded/content/dir")

# check the performance of the trained service, if training with new config, or for the first time
trained_service.evaluate_performance(eval_content_dir="downloaded/content/dir")

# create an image of the trained
trained_service.persist_trained_model(persist_dir="training/new_image_dir")
```

Note that the train and evaluation routine might take several hours on tens of thousands of documents.
See the Service's [technical docs](https://github.com/searchisko/project-classifier-poc/tree/master/deployable/technical_docs) for details.

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

The training text of documents is expected to be already preprocessed in csv files in a desired manner. 
The scored texts, however do not have to be preprocessed - the service will preprocess them before 
the scoring process using service default [preprocess_text(text)](https://github.com/searchisko/project-classifier-poc/tree/master/data/text_preprocess.py) 
method. See the method for more info.

Note that if using your own documents with your own preprocessing, the same preprocessing must be propagated into the service
so that the words are correctly matched.
This must be done by passing the preprocess method as argument into ``service.score_doc(preprocess_method=your_method(text))``.

Take a look into some implemented [data retrieval procedure](https://github.com/searchisko/project-classifier-poc/tree/master/data/searchisko_requestor.py)
to get inspired on how to perform a preprocess for your arbitrary data source.
