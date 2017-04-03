import json
from text_preprocess import preprocess_text
import requests

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class AccessDownloader:
    content_source_id = "api.access.redhat.com"
    url = 'https://api.access.redhat.com/rs/search'

    document_kinds = ["Article", "Solution"]

    project_name = None
    stemming = True

    header = ["sys_title", "sys_description", "sys_content_plaintext", "source"]

    sep = ","
    content_specific_attributes = {"sys_title": "publishedTitle",
                                   "sys_description": "abstract",
                                   "sys_content_plaintext": "body",
                                   "source": "source"}

    def __init__(self, project, csv_sep=",", drop_stemming=True):
        self.project_name = project
        self.sep = csv_sep
        self.stemming = drop_stemming

    # http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
    @staticmethod
    def replace_non_ascii(str):
        return ''.join([i if ord(i) < 128 else ' ' for i in str])

    @staticmethod
    def construct_query(filter_params):
        q_parts = []
        for k, v in filter_params.items():
            q_parts.append("%s:%s" % (k, v))

        return reduce(lambda x, y: x + "%20or%20" + y, q_parts)

    def json_to_csv(self, json_obj, content_type):
        out_csv = ""

        for entry in iter(json_obj):
            line = ""
            for att in map(lambda header_item: self.content_specific_attributes[header_item], self.header):
                if att in entry.keys():
                    if att == "body":
                        processed_text_unit = preprocess_text(entry[att][0], stemming=self.stemming)
                    else:
                        processed_text_unit = preprocess_text(entry[att], stemming=self.stemming)
                    processed_text = self.replace_non_ascii(processed_text_unit)
                    line += '"%s"%s' % (processed_text, self.sep)

                else:
                    # do not preprocess source identification
                    if att == "source":
                        line += '"%s"%s' % ("%s:%s" % (self.content_source_id, content_type), self.sep)
                    else:
                        line += '""%s' % self.sep
            line += '"%s"\n' % self.project_name
            yield line

    def download_and_parse(self, response_size=50, sample=None):
        if sample is None:
            sample = 100000
            # sample is to lower the retrieved documents if in test mode

        download_counter = 0
        # for document_kind in self.document_kinds:
        document_kind = "any"
        logging.info("Searching for document type: %s" % document_kind)
        offset = 0
        increase = response_size

        while increase >= response_size and offset < sample:
            params = [
                ('q', self.construct_query({'product': self.project_name})),
                ('rows', response_size),
                ('start', offset),
                ('fq', "language:en"),
                # ('fq', "documentKind:" + document_kind),
                ('sort', 'id asc')
            ]

            resp = requests.get(url=self.url, params=params)
            new_data = json.loads(resp.text)
            logging.info("Constructed url: %s" % resp.url)
            logging.info("Downloading %s/%s content (pulse %s)" % (download_counter + response_size,
                                                                   new_data["response"]["numFound"],
                                                                   response_size))

            for line in self.json_to_csv(new_data["response"]["docs"], document_kind):
                yield line

            increase = new_data["response"]["docs"].__len__()
            download_counter += increase
            offset += increase

        logging.info("Downloaded %s content items" % download_counter)
