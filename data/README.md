## Data

For a convenient collection of the fresh resources for the classified products, the 
[products_downloader.py](https://github.com/searchisko/project-classifier-poc/tree/master/data/products_downloader.py)
can be used to bulk-download and properly organize the known resources of RH products. 

Using the created **(classifier) conda environment**, you can do that by running the following:

```bash
cd data
# do not forget the suffix '/'
python products_downloader.py /abs/path/to/empty/directory/
```

This will create the last level of directory and download the indexed resources from the three linked sources: 
**access.redhat.com**, **dcp** sources bound to products, and **StackOverflow questions** 
from dcp with tags related to the products.

The products that are to be downloaded are listed in 
[product_list.py](https://github.com/searchisko/project-classifier-poc/tree/master/data/product_list.py)
file. Feel free to extend or limit the list here, if the linked sources contain new categories.

The download folder is then passed as parameter to training.

The downloaded sources might contain duplicates. Those are filtered during the training phase 
of the service.

The download process of all indexed content (appx. 60k documents) might take around **2 hours** 
depending mostly on speed of pre-processing.

A dataset of content irrelevant to any category is included in 
[content/none_category](https://github.com/searchisko/project-classifier-poc/blob/master/data/content/none_content)
directory. If you wish to include the **None category** in the training, copy a file **None_content.csv**
into the download dir given to products_downloader.py.

### Data format

The downloaded sources are categorized by product names into separate files for each category.
The naming convention of the files conveys the format necessary for training the service, as described in
[training](https://github.com/searchisko/project-classifier-poc/tree/master/search_service/training)
section: 

Each data file has a name in a form of:
**\<category\><suffix>.csv**, e.g. **\"eap_content.csv\"**.

The training csv files contains the **documents as rows** with the following attributes (colunms):
* **sys_title**: Document short header, if available. Used for training if **sys_description** is empty.
* **sys_description**: Document description, expected to be longer than **sys_title**. 
If empty, **sys_title** is used for training instead.
* **sys_content_plaintext**: Main piece of text of the document. Always used for training. Mandatory.
* **source**: Document source, e.g. stackoverflow. Can be left empty.
* **target**: Target category, matching the prefix of a file of this document.

