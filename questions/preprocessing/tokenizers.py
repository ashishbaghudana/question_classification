from abc import abstractmethod
from questions.structures.containers import Token

class Tokenizer(object):
    @abstractmethod
    def tokenize(self, string):
        pass


class SimpleSpaceTokenizer(object):
    """
    The Simple Space Tokenizer simply tokenizes the string on a space
    character. It assumes that the string has already been partly tokenized.

    An example would be : Who is the President of the US ?
    """
    DELIMITER = " "

    def tokenize(self, string):
        return string.split(SimpleSpaceTokenizer.DELIMITER)
