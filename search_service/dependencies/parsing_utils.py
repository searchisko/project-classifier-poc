import pandas as pd
import numpy as np

from text_preprocess import preprocess_text

from os import listdir
from os.path import isfile, join

from categorized_document import CategorizedDocument

import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# split tokenized text into all_sentences
def sentence_split(document):
    return filter(lambda sentence: len(sentence) > 0, document.split("."))


# split tokenized sentence (sequence of tokens separated by space) into tokens
def token_split(sentence):
    return filter(lambda token: len(token) > 0, sentence.split(" "))


def select_headers(df):
    selected_text_series = df.apply(lambda content: content["sys_title"] if content["sys_title"] else "", axis=1)\
                             .apply(lambda doc_text: doc_text.replace(".", " . "))\
                             .apply(lambda content: token_split(content))
    return selected_text_series


def select_training_content(df, make_document_mapping=False, sent_split=True):
    document_mapping_i = 0
    # relevant content attributes are sorted by relevance,
    # the considered content is from the first non-empty attribute

    document_mapping = []

    selected_content_container = []
    if sent_split:
        # considers optionally sys_content_plaintext if filled, or sys_description if not
        selected_text_series = df.apply(lambda content: sentence_split(content["sys_content_plaintext"])
                                        if content["sys_content_plaintext"] else
                                        sentence_split(content["sys_description"]) if content["sys_description"]
                                        else [content["sys_title"]],
                                        axis=1)
        for sentences in selected_text_series:
            # selected_content_container = np.append(selected_content_container, sentences)
            selected_content_container.extend(sentences)

            document_mapping.extend([document_mapping_i]*len(sentences))
            document_mapping_i += 1

        selected_content_container = pd.Series(selected_content_container)

    else:
        selected_content_container = df.apply(lambda content: content["sys_content_plaintext"]
                                              if content["sys_content_plaintext"] else
                                              content["sys_description"] if content["sys_description"]
                                              else content["sys_title"],
                                              axis=1)

    # treat dots as separate words, as used in demonstration
    selected_content_container.apply(lambda doc_text: doc_text.replace(".", " . "))

    # performance bottleneck supposedly here
    selected_content_container = pd.Series(map(lambda sentence: token_split(sentence), selected_content_container))
    if make_document_mapping:
        # document_mapping comes unfilled if whole documents mappings are returned
        return selected_content_container, pd.Series(document_mapping)
    else:
        return selected_content_container


def content_to_sentence_split(doc_content_plaintext):
    sentences = sentence_split(doc_content_plaintext)
    sen_token_split = map(lambda sentence: token_split(sentence), sentences)
    return sen_token_split


def sentence_list_from_content(content_df):
    doc_sentences = []
    content_df["content"].apply(lambda sentence: doc_sentences.append(sentence))

    return doc_sentences


def tagged_docs_from_content(content_series, content_headers, labels):
    content_df_with_index = pd.DataFrame(data={"content": content_series})
    content_df_with_index["index"] = np.arange(len(content_series))
    content_df_with_index["header"] = content_headers
    logging.debug("Initializing %s CategorizedDocuments" % len(content_df_with_index["index"]))

    return content_df_with_index.apply(lambda row: CategorizedDocument(row["content"],
                                                                       [row["index"]],
                                                                       labels.iloc[row["index"]],
                                                                       row["header"]), axis=1)


def tagged_docs_from_plaintext(content_series_plain, content_headers_plain, labels, preprocess_method=preprocess_text):
    content_series = pd.Series(content_series_plain).apply(preprocess_method).apply(token_split)
    content_headers = pd.Series(content_headers_plain).apply(preprocess_method).apply(token_split)

    return tagged_docs_from_content(content_series, content_headers, labels)


def create_dataset_from_tagged_docs(tagged_docs, directory, source=""):
    target_attributes = "sys_title,sys_description,sys_content_plaintext,source,target".split(",")
    target_df = pd.DataFrame(columns=target_attributes)

    target_df["sys_title"] = tagged_docs.apply(lambda cat_doc: cat_doc.header_words)
    target_df["sys_description"] = tagged_docs.apply(lambda cat_doc: "")
    target_df["sys_content_plaintext"] = tagged_docs.apply(lambda cat_doc: cat_doc.words)
    target_df["source"] = tagged_docs.apply(lambda cat_doc: source)
    target_df["target"] = tagged_docs.apply(lambda cat_doc: "None")
    with open(directory, "w") as wfile:
        target_df.to_csv(wfile, index=False, encoding='utf-8')


def parse_header_docs(full_docs):
    out_docs = full_docs.apply(lambda full_doc: CategorizedDocument(full_doc.header_words if not type(full_doc.header_words) == float else [],
                                                                    full_doc.tags,
                                                                    full_doc.category_expected,
                                                                    None))

    logging.debug("Initialized %s headers of %s for vectorization" % (len(out_docs), len(full_docs)))
    return out_docs


def content_from_words(word_list):
    if len(word_list) < 1:
        return ""
    return reduce(lambda x, y: "%s %s" % (x, y), word_list) + "."


def get_content_as_dataframe(content_basepath, basepath_suffix="_content.csv"):
    # initializes the vocabulary by the given categories (content_categories)
    # in the given directory (train_dir)
    content_categories = scan_directory_for_categories(content_basepath, basepath_suffix)

    all_content = pd.DataFrame(
        columns="sys_title,sys_description,source,sys_content_plaintext,target".split(","))

    # retrieve all content of all given categories
    for cat_label in content_categories:
        filepath = "%s/%s%s" % (content_basepath, cat_label, basepath_suffix)
        logging.info("Loading %s" % filepath)

        new_content = pd.read_csv(filepath, na_filter=False, error_bad_lines=False)
        all_content = all_content.append(new_content, ignore_index=True)

    return all_content


def drop_duplicate_docs(docs_series):
    docs_df = pd.DataFrame(columns=["header", "content"], index=docs_series.index)
    docs_df["header"] = docs_series.apply(lambda doc: doc.header_words).apply(str).apply(hash)
    docs_df["content"] = docs_series.apply(lambda doc: doc.words).apply(str).apply(hash)

    return docs_series[~docs_df.duplicated()]


def scan_directory_for_categories(train_dir, train_files_suffix="_content.csv"):
    dir_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
    return map(lambda dir_file_path: dir_file_path.replace(train_dir, "").replace(train_files_suffix, ""), dir_files)