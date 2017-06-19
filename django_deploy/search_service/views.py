import os
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from search_service import RelevanceSearchService

from datetime import datetime
import json
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# from .models import PageView

# Create your views here.
# models_dir = "/home/michal/Documents/Projects/ml/project-classifier-poc/project-classifier-poc/django_deploy/search_service/persisted_model_e20"
score_service_instance = RelevanceSearchService()


def _object_from_scores(doc_scores, scores_categories):
    scores_object = dict()
    for doc_id in doc_scores.index:
        scores_dict = dict()
        for cat in scores_categories:
            scores_dict[cat] = doc_scores.ix[doc_id, cat]
        scores_object[doc_id] = scores_dict

    return scores_object


@csrf_exempt
def score(request):
    start_time = datetime.utcnow()

    if request.method == "GET":
        # if GET request, parse required text parameters: doc_id, doc_header, doc_content
        try:
            doc_id = request.GET["doc_id"]
            doc_header = request.GET["doc_header"]
            doc_content = request.GET["doc_content"]
        except KeyError:
            return HttpResponse("ERROR: the service requires GET params: doc_id, doc_headber and doc_content", status=400)

        logging.info("GET: Scoring document %s: header len: %s, content len: %s"
                     % (doc_id, len(doc_header.split()), len(doc_content.split())))

        doc_scores, scores_categories = score_service_instance.score_doc(doc_id, doc_header, doc_content)

    elif request.method == "POST":
        try:
            doc_id = request.POST["doc_id"]
            doc_header = request.POST["doc_header"]
            doc_content = request.POST["doc_content"]
        except KeyError:
            return HttpResponse("ERROR: the service requires GET params: doc_id, doc_headber and doc_content", status=400)

        logging.info("POST: Scoring document %s: header len: %s, content len: %s"
                     % (doc_id, len(doc_header.split()), len(doc_content.split())))

        doc_scores, scores_categories = score_service_instance.score_doc(doc_id, doc_header, doc_content)
    else:
        return HttpResponse("The service responds on HTTP GET and POST", status=400)

    scores_object = _object_from_scores(doc_scores, scores_categories)
    response_object = {"sys_meta": {"response_status": "OK",
                                    "request_time": start_time.isoformat(),
                                    "response_time": datetime.utcnow().isoformat(),
                                    "sys_model_training_time": score_service_instance.service_meta["model_train_end_timestamp"].isoformat()
                                    },
                       "docs_scoring_results": scores_object}

    response_json = json.dumps(response_object)

    return HttpResponse(str(response_json), content_type="application/json", status=200)


def score_bulk(request):
    return score(request)
