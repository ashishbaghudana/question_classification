from abc import abstractmethod
from questions.structures.containers import Token
from nltk import word_tokenize

class Tokenizer(object):
    @abstractmethod
    def tokenize(self, string):
        pass


class SimpleSpaceTokenizer(Tokenizer):
    """
    The Simple Space Tokenizer simply tokenizes the string on a space
    character. It assumes that the string has already been partly tokenized.

    An example would be : Who is the President of the US ?
    """
    DELIMITER = " "

    def tokenize(self, string):
        return string.split(SimpleSpaceTokenizer.DELIMITER)


class NLTKTokenizer(Tokenizer):
    """
    The NLTK Tokenizer is trained on the punkt corpus and matches the training
    data's tokenized style
    """
    def tokenize(self, string):
        return word_tokenize(string)
