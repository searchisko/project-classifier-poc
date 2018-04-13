## Deployment

First, go to 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/views.py) 
and see where to set the path to a directory with pre-trained
image of the service.

After the service is correctly linked to the trained image in 
[views.py](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/views.py),
We need to have installed all the dependent libraries in the **deployment environment** 
using ``pip install -r requirements.txt``.

Then, we are ready to start the django servlet, with either:
 
``./manage.py runserver`` or 

``gunicorn wsgi:application``

If the server has loaded with no errors, the application is ready to respond on a given address.
Try to request a root to see if the service is running.

After that, go ahead and try to feed the service REST API with some dummy text for scoring:

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

Note the linked service image is loaded with the first request to the API, so the first request expects to take
1 - 5 seconds, depending on the hardware and model complexity.

### REST API
After everything looks good, take a look at a full 
[REST API Documentation](https://github.com/searchisko/project-classifier-poc/blob/master/search_service/API)
.