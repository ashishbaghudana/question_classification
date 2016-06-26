import copy
import random

class Dataset(object):
    """
    The dataset holds all the questions with their descriptions
    """
    def __init__(self):
        self.questions = []

    def add(self, question):
        self.questions.append(question)

    def __iter__(self):
        for question in self.questions:
            yield question

    def __len__(self):
        return len(self.questions)

    def n_fold_split(self, n=5, fold_nr=None):
        documents = copy.deepcopy(self.questions)
        random.seed(2727)
        random.shuffle(documents)

        fold_size = int(len(documents) / n)

        def get_fold(fold_nr):
            """
            fold_nr starts in 0
            """
            start = fold_nr * fold_size
            end = start + fold_size
            test_docs = documents[start : end]
            train_docs = [doc for doc in documents if doc not in test_docs]

            test = Dataset()
            for doc in test_docs:
                test.add(doc)

            train = Dataset()
            for doc in train_docs:
                train.add(doc)

            return train, test

        if fold_nr:
            assert(0 <= fold_nr < n)
            yield get_fold(fold_nr)
        else:
            for fold_nr in range(n):
                yield get_fold(fold_nr)

    def target(self):
        classes = []
        for question in self:
            classes.append(question.type)
        return classes

    def questions_as_array(self):
        questions = []
        for question in self:
            questions.append(question.text)
        return questions

class Question(object):
    """
    The question object holds all the information related to a question
    """

    def __init__(self, question_text, question_type=None, tokens=[],
            features={}):
        self.text       = question_text
        self.type       = question_type
        self.tokens     = tokens
        self.features   = features

    def get_text(self):
        return self.text

    def set_text(self, text):
        self.text = text

    def get_type(self):
        return self.type

    def set_type(self, question_type):
        self.type = question_type

    def get_tokens(self):
        for token in self.tokens:
            yield token

    def add_token(self, token):
        self.tokens.append(token)

    def add_feature(self, key, value):
        self.features[key] = value

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __contains__(self, item):
        return item in self.text.lower()

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

class Token(object):
    """
    The token class represents each token in a sentence
    """
    def __init__(self, token_text, token_id, features={}):
        self.id         = token_id
        self.text       = token_text
        self.features   = features

    def add_feature(self, key, value):
        self.features[key] = value

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text
