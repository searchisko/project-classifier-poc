import pandas as pd
import numpy as np

from collections import namedtuple

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


CategorizedDocument = namedtuple('CategorizedDocument', 'words tags category_expected header_words')


# split tokenized text into all_sentences
def sentence_split(document):
    return filter(lambda sentence: len(sentence) > 0, document.split("."))


# split tokenized sentence (sequence of tokens separated by space) into tokens
def token_split(sentence):
    return filter(lambda token: len(token) > 0, sentence.split(" "))

training_attributes = ["sys_content_plaintext", "sys_description", "sys_title"]


def select_headers(df):
    return df[training_attributes[2]].apply(lambda doc_text: doc_text.replace(".", " . "))\
        .apply(lambda content: token_split(content))


def select_training_content(df, make_document_mapping=False, sent_split=True):
    document_mapping_i = 0
    # relevant content attributes are sorted by relevance,
    # the considered content is from the first non-empty attribute

    document_mapping = []

    selected_content_container = []
    if sent_split:
        # considers optionally sys_content_plaintext if filled, or sys_description if not
        selected_text_series = df.apply(lambda content: sentence_split(content[training_attributes[0]])
                                        if content[training_attributes[0]] else
                                        sentence_split(content[training_attributes[1]]) if content[training_attributes[1]]
                                        else [content[training_attributes[2]]],
                                        axis=1)
        for sentences in selected_text_series:
            # selected_content_container = np.append(selected_content_container, sentences)
            selected_content_container.extend(sentences)

            document_mapping.extend([document_mapping_i]*len(sentences))
            document_mapping_i += 1

        selected_content_container = pd.Series(selected_content_container)

    else:
        selected_content_container = df.apply(lambda content: content[training_attributes[0]]
                                              if content[training_attributes[0]] else
                                              content[training_attributes[1]] if content[training_attributes[1]]
                                              else content[training_attributes[2]],
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
    logging.info("Initializing %s CategorizedDocuments" % len(content_df_with_index["index"]))

    return content_df_with_index.apply(lambda row: CategorizedDocument(row["content"],
                                                                       [row["index"]],
                                                                       labels.iloc[row["index"]],
                                                                       row["header"]), axis=1)


def parse_header_docs(full_docs):
    out_docs = full_docs.apply(lambda full_doc: CategorizedDocument(full_doc.header_words if not type(full_doc.header_words) == float else [],
                                                                    full_doc.tags,
                                                                    full_doc.category_expected,
                                                                    None))

    logging.info("Initialized %s headers of %s for vectorization" % (len(out_docs), len(full_docs)))
    return out_docs


def content_from_words(word_list):
    if len(word_list) < 1:
        return ""
    return reduce(lambda x, y: "%s %s" % (x, y), word_list) + "."


def get_content_as_dataframe(content_basepath, basepath_suffix, content_categories):
    # retrieve all content of all given categories
    all_content = pd.DataFrame(
        columns="sys_title,sys_description,source,sys_content_plaintext,target".split(","))
    for cat_label in content_categories:
        new_content = pd.read_csv("%s/%s%s" % (content_basepath, cat_label, basepath_suffix),
                                  na_filter=False, error_bad_lines=False)
        all_content = all_content.append(new_content, ignore_index=True)

    return all_content
