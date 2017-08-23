## Data

For a convenient collection of the fresh resources for the classified products, the 
[products_downloader.py](https://github.com/searchisko/project-classifier-poc/tree/master/data/products_downloader.py)
can be used to bulk-download and properly organize the known resources of RH products:

```python
from data.products_downloader import download
download(download_dir="/abs/path/to/empty/directory")
```

This will download the indexed resources from the three linked sources: **access.redhat.com**, **dcp** sources
bound to products, and **StackOverflow questions** from dcp with tags related to the products.

The downloaded sources might contain duplicates. Those are filtered during the training phase 
of the service.

The download process of all indexed content (appx. 60k documents) might take around **2 hours**.

### Data format

The downloaded sources are categorized by product names into separate files for each category.
The naming convention of the files conveys the format necessary for training the service, as described in
[training](https://github.com/searchisko/project-classifier-poc/tree/master/deployable/search_service/training)
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

