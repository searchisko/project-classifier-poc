import json
import requests
import sys
from text_preprocess import preprocess_text
import math

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

url = 'http://dcp2.jboss.org/v2/rest/search'
response_size = 50
increase = response_size
data = []
offset = 0
sample = 100000

project_name = "fsw"
out_file = "content/%s_content.csv" % project_name

sep = ","

gathered_attributes = ["sys_title", "sys_description", "source", "sys_content_plaintext"]
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
                if att == "source":
                    line += '"%s"%s' % (e_source[att], sep)
                else:
                    processed_text_unit = preprocess_text(e_source[att])
                    processed_text = replace_non_ascii(processed_text_unit)
                    line += '"%s"%s' % (processed_text, sep)
            else:
                line += '""%s' % sep
        line += '"%s"\n' % target_attribute
        out_csv += line

    return out_csv

header_fields = gathered_attributes + ["target"]
download_counter = 0
with open(out_file, "a+") as out_file:
    # aggregate header to a single string and write it to the output file
    out_file.write("%s\n" % reduce(lambda current, update: "%s%s%s" % (current, sep, update), header_fields))

    while increase >= response_size and offset < sample:
        params = {
            'size': response_size,
            'field': '_source',
            'project': project_name,
            'from': offset
        }

        resp = requests.get(url=url, params=params)
        new_data = json.loads(resp.text)

        logging.info("Downloading %s/%s content (pulse %s)" % (download_counter+response_size,
                                                               new_data["hits"]["total"],
                                                               response_size))

        csv = json_to_csv(new_data["hits"]["hits"])
        out_file.write(csv)

        increase = len(new_data["hits"]["hits"])
        download_counter += increase
        offset += increase

logging.info("Downloaded %s content items" % download_counter)
