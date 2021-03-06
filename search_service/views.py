from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from search_service import ScoringService

from datetime import datetime
import json
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

generic_error_message = "</br>See <a href='https://github.com/searchisko/project-classifier-poc'>" \
                        "https://github.com/searchisko/project-classifier-poc</a> README for documentation."

# TODO: set relative path to the trained image here,
# or set an absolute path to score_service_instance.service_image_dir
score_service_instance = ScoringService(image_dir="trained_service_prod")


def _object_from_scores(doc_scores, scores_categories):
    scores_object = dict()
    for doc_id in doc_scores.index:
        scores_dict = dict()
        for cat in scores_categories:
            scores_dict[cat] = doc_scores.ix[doc_id, cat]
        scores_object[doc_id] = scores_dict

    return scores_object


def _verify_request(request):
    if request.method != "POST":
        return "The service responds on HTTP POST. " + generic_error_message

    return None


def _get_sys_meta(request_time):
    sys_meta_object = {"response_status": "OK",
                       "request_time": str(request_time.isoformat()),
                       "response_time": str(datetime.utcnow().isoformat()),
                       "sys_service_meta": str(score_service_instance.service_meta["model_train_end_timestamp"]),
                       "sys_requests_counter": score_service_instance.service_meta["score_requests_counter"],
                       }
    if score_service_instance.service_meta["model_eval_result"] is not None:
        sys_meta_object["sys_negative_performance"] = \
            score_service_instance.service_meta["model_eval_result"]["mean_negative_performance"]
        sys_meta_object["sys_positive_performance"] = \
            score_service_instance.service_meta["model_eval_result"]["mean_positive_performance"]
        sys_meta_object["sys_cats_performance"] = \
            score_service_instance.service_meta["model_eval_result"]["categories_mean_performance"].to_dict()

    return sys_meta_object


@csrf_exempt
def service_root(_):
    display_meta = score_service_instance.service_meta["model_train_start_timestamp"]
    display_meta = display_meta if display_meta else "Not loaded yet"
    resp_html = """
    <b>Up and running!</b></n></br>
    Classification service is ready to use with model version: <span class='id'>%s</span></br>
    Check out the service
    <a href='https://github.com/searchisko/project-classifier-poc/tree/master/search_service/API'>
    REST API documentation</a>.
    """ % display_meta

    return HttpResponse(resp_html)


@csrf_exempt
def score(request):
    request_time = datetime.utcnow()

    if _verify_request(request) is not None:
        return HttpResponse(_verify_request(request), status=400)

    try:
        req_text = request.read()
        logging.debug(req_text)
        request_json = json.loads(req_text)
    except ValueError:
        return HttpResponse("The requested json is in malformed format. Please check. \n" + generic_error_message,
                            status=400)

    # logging.debug("POST on score(), body:\n%s" % request_json)
    try:
        doc_id = request_json["doc"]["id"]
        doc_title = request_json["doc"]["title"]
        doc_content = request_json["doc"]["content"]
    except KeyError:
        return HttpResponse("Wrong request json format. " + generic_error_message, status=400)

    try:
        sys_meta = request_json["sys_meta"]
    except KeyError:
        sys_meta = False

    try:
        doc_scores = score_service_instance.score_doc(doc_id, doc_title, doc_content)
    except Exception as e:
        logging.error("POST on score(): Scoring doc %s terminated the scoring with error: \n%s" % (doc_id, e))
        raise

    scores_categories = doc_scores.columns.values

    scores_object = _object_from_scores(doc_scores, scores_categories)
    single_doc_scores = scores_object[scores_object.keys()[0]]

    response_object = {"scoring": single_doc_scores}
    if sys_meta:
        response_object["sys_meta"] = _get_sys_meta(request_time)

    response_json = json.dumps(response_object)
    response_time = (datetime.utcnow() - request_time).total_seconds()

    logging.info("POST on score(): Scoring: %s, response_time: %s, header_len: %s, content_len: %s"
                 % (doc_id, response_time, len(doc_title.split()), len(doc_content.split())))

    return HttpResponse(str(response_json), content_type="application/json", status=200)


@csrf_exempt
def score_bulk(request):
    request_time = datetime.utcnow()

    if _verify_request(request) is not None:
        return HttpResponse(_verify_request(request), status=400)
    try:
        request_json = json.loads(request.read())
    except ValueError:
        return HttpResponse("The requested json is in malformed format. Please check. \n" + generic_error_message,
                            status=400)

    try:
        doc_ids = request_json["docs"].keys()
        doc_titles = map(lambda id: request_json["docs"][id]["title"], doc_ids)
        doc_contents = map(lambda id: request_json["docs"][id]["content"], doc_ids)
    except KeyError:
        return HttpResponse("Wrong request json format. " + generic_error_message, status=400)

    try:
        sys_meta = request_json["sys_meta"]
    except KeyError:
        sys_meta = False

    try:
        doc_scores = score_service_instance.score_docs_bulk(doc_ids, doc_titles, doc_contents)
    except Exception as e:
        logging.error("POST on scoreBulk(): Scoring docs %s terminated the service with error: \n%s" % (doc_ids, e))
        raise

    scores_categories = doc_scores.columns.values
    scores_object = _object_from_scores(doc_scores, scores_categories)

    response_object = {"scoring": scores_object}

    if sys_meta:
        response_object["sys_meta"] = _get_sys_meta(request_time)

    response_json = json.dumps(response_object)
    response_time = (datetime.utcnow() - request_time).total_seconds()

    logging.info("POST on scoreBulk(): Scoring: %s docs, response_time: %s, summed_response_time: %s, "
                 "header_len: %s, content_len: %s"
                 % (len(doc_ids), response_time / len(doc_ids), response_time,
                    sum(map(len, map(lambda s: str(s).split(), doc_titles))) / float(len(doc_ids)),
                    sum(map(len, map(lambda s: str(s).split(), doc_contents))) / float(len(doc_ids))))

    return HttpResponse(str(response_json), content_type="application/json", status=200)
