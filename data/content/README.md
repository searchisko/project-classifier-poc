## Training and eval content

The directory contains: 

1. **[/prod](https://github.com/searchisko/project-classifier-poc/blob/master/data/content/prod)** 
directory: resources of the **classified categories** 
as well as the randomly-sampled **None-category** content of 7500 docs from StackOverflow,
New York Times and Twitter. The resources has removed stopwords and numbers, persisted stemming.
These resources were used for training the **production service** model.

3. **[/experimental](https://github.com/searchisko/project-classifier-poc/blob/master/data/content/experimental)** 
directory with sets of irrelevant documents used for performance evaluation 
on negative data set (in analyses and when estimating the categories threshold).

4. **[/books_test](https://github.com/searchisko/project-classifier-poc/blob/master/data/content/books_test)** 
directory with the preprocessed data set of books 
from **developer.redhat.com/resources** used as experimental sample for classification of unclassified
content from developer.redhat resources, that the service is eventually expected to score and search in.

You can find the full relevant data set downloaded using the 
[products_downloader](https://github.com/searchisko/project-classifier-poc/blob/master/data/products_downloader.py)
zipped in 
[here](https://drive.google.com/open?id=0B_42L5-Ve7j2d3pTQm5kaUhlM28).