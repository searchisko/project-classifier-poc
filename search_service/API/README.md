## Search service ::  REST API
The running django application responds **POST** requests having JSON body 
in a described format on two adresses:

1a) **/score**: **request** scoring the relevance of a single document:

```json
{
	"sys_meta": false,
	"doc": {
		"id": "DOC_123",
		"title": "One smart doc",
		"content": "This document has a lot of funny stuff."
	}
}
```

1b) **/score** **responds** in a form:

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

2a) **/scoreBulk**: **request** scoring the a bulk of documents at once. The method is much faster when scoring bigger bunch 
of docs at once, than :

```json
{
	"sys_meta": true,
	"docs": {
		"DOC_123": {
		    "title": "One smart doc",
		    "content": "This document has a lot of funny stuff."
		},
		"DOC_234": {
		    "title": "One silly doc",
		    "content": "This doc is not as funny as the first one."
		}
	}
}

```

2b) **/scoreBulk** **responds** in a form:

```json
{
    "scoring": {
        "DOC_123": {
            "softwarecollections": 0.000064274845465339681,
            "brms": 0.0009433698353256692,
            "...": "..."
        },
        "DOC_234": {
            "softwarecollections": 0.000032687001843056951,
            "brms": 0.002657126406253485,
            "...": "..."
        }
    },
    "sys_meta": {
        "response_status": "OK",
        "request_time": "2017-06-22T13:38:07.666230",
        "response_time": "2017-06-22T13:38:07.723830",
        "sys_model_training_time": "2017-06-15T12:55:01.981282"
    }
}
```

**Note** that **sys_meta** object is included in a response if ```"sys_meta": true``` is set in request. 
It defaults to ```false```.
