import json
from text_preprocess import preprocess_text
import requests

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SearchiskoDownloader:
    APPEND = False
    url = 'http://dcp2.jboss.org/v2/rest/search'
    project_name = None
    stemming = True

    sep = ","

    gathered_attributes = ["sys_title", "sys_description", "sys_content_plaintext", "source"]

    def __init__(self, project, csv_sep=",", drop_stemming=True):
        self.project_name = project
        self.sep = csv_sep
        self.stemming = drop_stemming

    # http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
    @staticmethod
    def replace_non_ascii(str):
        return ''.join([i if ord(i) < 128 else ' ' for i in str])

    def json_to_csv(self, json_obj):
        out_csv = ""

        for entry in json_obj:
            e_source = entry["_source"]
            line = ""
            for att in self.gathered_attributes:
                if att in e_source.keys():
                    # do not preprocess source identification
                    if att == "source":
                        line += '"%s"%s' % (e_source[att], self.sep)
                    else:
                        processed_text_unit = preprocess_text(e_source[att], stemming=self.stemming)
                        processed_text = self.replace_non_ascii(processed_text_unit)
                        line += '"%s"%s' % (processed_text, self.sep)
                else:
                    line += '""%s' % self.sep
            line += '"%s"\n' % self.project_name
            yield line

    def download_and_parse(self, response_size=50, sample=None):
        download_counter = 0
        offset = 0
        increase = response_size

        if sample is None:
            sample = 100000
            # sample is to lower the retrieved documents if in test mode

        while increase >= response_size and offset < sample:
            params = {
                'size': response_size,
                'field': '_source',
                'project': self.project_name,
                'from': offset
            }

            resp = requests.get(url=self.url, params=params)
            new_data = json.loads(resp.text)
            logging.info("Constructed url: %s" % resp.url)
            logging.info("Downloading %s/%s content (pulse %s)" % (download_counter+response_size,
                                                                   new_data["hits"]["total"],
                                                                   response_size))
            for line in self.json_to_csv(new_data["hits"]["hits"]):
                yield line

            increase = len(new_data["hits"]["hits"])
            download_counter += increase
            offset += increase

        logging.info("Downloaded %s content items" % download_counter)
