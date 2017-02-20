import json
from text_preprocess import preprocess_text
import requests

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

APPEND = False

url = 'http://dcp2.jboss.org/v2/rest/search'
response_size = 50
increase = response_size
data = []
offset = 0
sample = 100000

project_name = "eap"
out_file = "content/playground/%s_nostem.csv" % project_name

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
                if att == "source":
                    line += '"%s"%s' % (e_source[att], sep)
                else:
                    processed_text_unit = preprocess_text(e_source[att], stemming=False)
                    processed_text = replace_non_ascii(processed_text_unit)
                    line += '"%s"%s' % (processed_text, sep)
            else:
                line += '""%s' % sep
        line += '"%s"\n' % target_attribute
        out_csv += line

    return out_csv

header_fields = gathered_attributes + ["target"]
file_handle_type = "a+" if APPEND else "w"
download_counter = 0
with open(out_file, file_handle_type) as out_file:
    if not APPEND:
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
