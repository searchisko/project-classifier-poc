from data.searchisko_requestor import SearchiskoDownloader
from data.searchisko_so_requestor import StackOverflowDownloader
from data.access_redhat_requestor import AccessDownloader
from data.product_list import product_list

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

downloader_instances = [AccessDownloader, SearchiskoDownloader, StackOverflowDownloader]

# TODO: fill in absolute path for download content output directory
output_path = "/home/michal/Documents/Projects/ml/project-classifier-poc/project-classifier-poc/data/content/prod_no_preproc/"

csv_sep = ","


# http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
def replace_non_ascii(str):
    return ''.join([i if ord(i) < 128 else ' ' for i in str])


# used to drop the blank lines in the content that destroy the csv format
def drop_spaces(text):
    text_cleared = replace_non_ascii(text).replace('"', "")
    return " ".join(text_cleared.split())


def download(download_dir):
    content_files_pattern = "%s_content.csv"
    content_path = download_dir + content_files_pattern

    logging.info("Started crawling of %s categories: \n%s" % (len(product_list), product_list))
    logging.info("Output to be written to: %s" % content_path)

    for category in product_list:
        logging.info("Downloading category %s" % category)
        with open(content_path % category, mode="w") as out_cat_file:
            # header write
            header_fields = ["sys_title", "sys_description", "sys_content_plaintext", "source", "target"]
            # aggregate header to a single string and write it to the output file
            out_cat_file.write("%s\n" % reduce(lambda current, update: "%s%s%s" % (current, csv_sep, update), header_fields))

            for downloader_class in downloader_instances:
                logging.info("Using %s downloader" % downloader_class)

                downloader = downloader_class(project=category, csv_sep=csv_sep,
                                              drop_stemming=True, preprocessor=drop_spaces)
                # use .download_and_parse(sample=60) param to limit content for content format testing
                download_generator = downloader.download_and_parse()
                for line in download_generator:
                    out_cat_file.write(line)

download(download_dir=output_path)
