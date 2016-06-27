from questions.structures.containers import Dataset, Question, Token

class AnnotationReader(object):
    """
    The dataset is of the form:

    what    what is the best ebook reader ?
    when    when does the train arrive ?
    ...

    The dataset is present in a raw text format that can be read in directly
    """

    DELIMITER = '\t'

    def __init__(self, input_file, tokenizer):
        self.input_file = input_file
        self.tokenizer = tokenizer

    def parse(self):
        dataset = Dataset()
        with open(self.input_file) as file_reader:
            for line in file_reader:
                q_type, q_text = line.split(AnnotationReader.DELIMITER)
                tokens = [Token(token_text, token_id) for token_id, token_text
                            in enumerate(self.tokenizer.tokenize(q_text.strip()))]
                question = Question(q_text.strip(), q_type, tokens=tokens)
                dataset.add(question)
        return dataset


class CommandLineInterfaceReader(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def parse(self, text):
        dataset = Dataset()
        tokens = [Token(token_text, token_id) for token_id, token_text
                    in enumerate(self.tokenizer.tokenize(text.strip()))]
        question = Question(text.strip(), tokens=tokens)
        dataset.add(question)
        return dataset



class InputReader(object):
    """
    The dataset is of the form?

    what is the best ebook reader?
    when does the train arrive?
    ...

    The dataset is present in a raw text format without classes
    """

    def __init__(self, input_file):
        self.input_file = input_file

    def parse(self):
        dataset = Dataset()
        with open(self.input_file) as file_reader:
            for question_text in file_reader:
                question = Question(question_text)
                dataset.add(question)
        return dataset
