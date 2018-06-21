## Pretrained services

These images are all trained on apx. 50 000 documents, from [/data/content/prod](https://github.com/searchisko/project-classifier-poc/tree/master/data/content/prod) folder.
in addition to the relevant (products') documents, it contains a "None" category 
of 7500 non-relevant documents from StackOverflow, New York Times articles, and Political tweets.

1. (deprecated) Image (created 25.7.2017) for **0.8 version of Gensim**, 
trained on **headers** exclusively selected from **sys_title**: 
[https://drive.google.com/file/d/1bnN-hjtQxa1zmxGgBpwtr-6g9wIL0d9E/view](https://drive.google.com/file/d/1bnN-hjtQxa1zmxGgBpwtr-6g9wIL0d9E/view).

2. (deprecated) Image (created 25.7.2017) for **0.8 version of Gensim**, 
with headers trained on **sys_description, if present**, otherwise on sys_title: 
[https://drive.google.com/file/d/0B_42L5-Ve7j2STFodkprVUY0Wms/view](https://drive.google.com/file/d/0B_42L5-Ve7j2STFodkprVUY0Wms/view).

3. Image (created 14.6.2018) for **3.4.0 version of Gensim**, 
trained on **headers** exclusively selected from **sys_title**: 
[https://drive.google.com/file/d/1VpBcaes2spsPhLTm0Guhce-GcgccZp3u/view](https://drive.google.com/file/d/1VpBcaes2spsPhLTm0Guhce-GcgccZp3u/view).

4. Image (created 14.6.2018) for **3.4.0 version of Gensim**, 
with headers trained on **sys_description, if present**, otherwise on sys_title: 
[https://drive.google.com/file/d/14oIG20okeL-03Hxduq0f9KVxOp0HnJ_D/view](https://drive.google.com/file/d/14oIG20okeL-03Hxduq0f9KVxOp0HnJ_D/view).

The service image contains the models of scoring tuner, Gensim's doc2vec models, and sklearn's classifier. 
These modules can be potentially independently trained and seamlessly changed in the image directory, 
however it is not the common use case.

To make the service to use one of these pre-trained images, collect all the files into the **same folder** and 
do not forget to **reference the folder in [views.py](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/views.py)**
