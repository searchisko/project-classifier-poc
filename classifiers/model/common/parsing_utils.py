import pandas as pd
from gensim.models import doc2vec

from collections import namedtuple

CategorizedDocument = namedtuple('CategorizedDocument', 'words tags category_expected')


# split tokenized text into all_sentences
def sentence_split(document):
    return filter(lambda sentence: len(sentence) > 0, document.split("."))


# split tokenized sentence (sequence of tokens separated by space) into tokens
def token_split(sentence):
    return filter(lambda token: len(token) > 0, sentence.split(" "))


def select_training_content(df, make_document_mapping=False, sent_split=True):
    document_mapping_i = 0
    # relevant content attributes are sorted by relevance,
    # the considered content is from the first non-empty attribute

    training_attributes = ["sys_content_plaintext", "sys_description", "sys_title"]
    document_mapping = []

    selected_content_container = []
    if sent_split:
        # considers optionally sys_content_plaintext if filled, or sys_description if not
        selected_text_series = df.apply(lambda content: sentence_split(content[training_attributes[0]])
                                        if content[training_attributes[0]] else
                                        sentence_split(content[training_attributes[1]]) if content[training_attributes[1]]
                                        else content[training_attributes[2]],
                                        axis=1)
        for sentences in selected_text_series:
            # selected_content_container = np.append(selected_content_container, sentences)
            selected_content_container.extend(sentences)

            document_mapping.extend([document_mapping_i]*len(sentences))
            document_mapping_i += 1

    else:
        selected_content_container = df.apply(lambda content: content[training_attributes[0]]
                                              if content[training_attributes[0]] else
                                              content[training_attributes[1]] if content[training_attributes[1]]
                                              else content[training_attributes[2]],
                                              axis=1)

    # treat dots as separate words, as used in demonstration
    selected_content_container.apply(lambda doc_text: doc_text.replace(".", " . "))

    # TODO: performance bottleneck supposedly here
    selected_content_container = pd.Series(map(lambda sentence: token_split(sentence), selected_content_container))
    if make_document_mapping:
        # document_mapping comes unfilled if whole documents mappings are returned
        return selected_content_container, pd.Series(document_mapping)
    else:
        return selected_content_container


# not currently used
# def csv_to_document(doc_df, category=None):
#     # doc_df = pd.read_csv(csv_content, header=attributes_format)
#     content = select_training_content(doc_df)
#
#     # TODO: this might be extended with other metadata carried over the classification
#     doc_metadata = {"doc_title": doc_df["sys_title"],
#                     "doc_category": category}
#
#     document = Document(plain_text=content, attributes=doc_metadata)
#
#     return document


def content_to_sentence_split(doc_content_plaintext):
    sentences = sentence_split(doc_content_plaintext)
    sen_token_split = map(lambda sentence: token_split(sentence), sentences)
    return sen_token_split


def sentence_list_from_content(content_df):
    doc_sentences = []
    content_df["content"].apply(lambda sentence: doc_sentences.append(sentence))

    return doc_sentences


def tagged_docs_from_content(content_series, labels):
    content_df_with_index = pd.DataFrame(data=content_series, columns=["content"])
    content_df_with_index["index"] = content_series.index.values
    return content_df_with_index.apply(lambda row: CategorizedDocument(row["content"],
                                                                       [row["index"]],
                                                                       labels[row["index"]]), axis=1)
