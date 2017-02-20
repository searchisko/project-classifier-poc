import json
import logging

import requests

import so_tag_mapping
from text_preprocess import preprocess_text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# TODO: http://dcp2.jboss.org/v2/rest/search?size=50&field=_source&tag=rhel&type=stackoverflow_question

APPEND = True

url = 'http://dcp2.jboss.org/v2/rest/search'
data = []

project_name = "eap"
out_file = "content/playground/%s_so_content.csv" % project_name

sep = ","

gathered_attributes = ["sys_title", "sys_description", "sys_content_plaintext", "source"]
target_attribute = project_name


# http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
def replace_non_ascii(str):
    return ''.join([i if ord(i) < 128 else ' ' for i in str])


def json_to_csv(json_obj):
    out_csv = ""

    for entry in json_obj:
        e_source = entry["_source"]
        line = ""
        for att in gathered_attributes:
            if att in e_source.keys():
                # do not preprocess source identification
                if att in ["source"]:
                    line += '"%s"%s' % (e_source[att], sep)
                else:
                    processed_text_unit = preprocess_text(e_source[att], stemming=True)
                    processed_text = replace_non_ascii(processed_text_unit)
                    line += '"%s"%s' % (processed_text, sep)
            else:
                line += '""%s' % sep
        # print empty description tag for compatibility with std parser
        # print target category for category training / evaluation
        line += '"%s"\n' % target_attribute

        out_csv += line

    return out_csv


header_fields = gathered_attributes + ["target"]
file_handle_type = "a+" if APPEND else "w"
with open(out_file, file_handle_type) as out_file:
    if not APPEND:
        # aggregate header to a single string and write it to the output file
        out_file.write("%s\n" % reduce(lambda current, update: "%s%s%s" % (current, sep, update), header_fields))

    # retrieve product tags from mapping as retrieved from:
    # https://developers.redhat.com/sites/default/files/js/js_fP8gNSfNygBRHdDsIOIxFrpv92iS6fyy9Gogv03CC-U.js
    try:
        product_tags = so_tag_mapping.project_tag_mapping[project_name]["stackoverflow"]
        if product_tags is None:
            product_tags = []
    except KeyError:
        product_tags = []

    logging.info("Found %s tags for project %s" % (product_tags, project_name))

    download_counter = 0
    for current_tag in product_tags:
        # crawling params configuration is reset after every tag iteration
        offset = 0
        sample = 100000
        response_size = 50
        increase = response_size

        while increase >= response_size and offset < sample:
            params = {
                'size': response_size,
                'field': '_source',
                'from': offset,
                'type': 'stackoverflow_question',
                'tag': current_tag
            }

            resp = requests.get(url=url, params=params)
            new_data = json.loads(resp.text)

            logging.info("Downloaded %s/%s content of '%s' tag (pulse %s)" % (download_counter,
                                                                             new_data["hits"]["total"],
                                                                             current_tag,
                                                                             response_size))

            csv = json_to_csv(new_data["hits"]["hits"])
            out_file.write(csv)

            increase = len(new_data["hits"]["hits"])
            download_counter += increase
            offset += increase

logging.info("Done. Downloaded %s content items" % download_counter)
