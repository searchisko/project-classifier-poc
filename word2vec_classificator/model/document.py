import pandas as pd


# to-be-classified (not training) document instance
# not currently used - redundantly complicates the Object model - will likely be removed from in the future
class Document:

    attributes = dict()
    plain_text = None
    sentence_list = []

    def __init__(self, plain_text, attributes):
        self.attributes = attributes
        self.sentence_list = self._preprocess_text(plain_text)

    # split tokenized text into all_sentences
    @staticmethod
    def _sentence_split(document):
        return filter(lambda sentence: len(sentence) > 0, document.split("."))

    # split tokenized sentence (sequence of tokens separated by space) into tokens
    @staticmethod
    def _token_split(sentence):
        return filter(lambda token: len(token) > 0, sentence.split(" "))

    def _preprocess_text(self, plain_text):
        sentence_list = self._sentence_split(plain_text)
        sen_word_list = pd.Series(map(lambda sentence: self._token_split(sentence), sentence_list))
        return sen_word_list
