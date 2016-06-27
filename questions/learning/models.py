from abc import abstractmethod
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from questions.utils import constants
import logging
import gensim
import numpy as np

class Model(object):
    """
    Abstract Class for generating different training models
    """
    @abstractmethod
    def train(self, train, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, test, model=None):
        pass


class Evaluator(object):
    """
    Encapsulate all the evaluation metrics for the different trained models
    """
    def __init__(self, labels=None):
        self.labels = labels

    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred, strategy='weighted'):
        return metrics.precision_score(y_true, y_pred, labels=self.labels,
            average=strategy)

    def recall(self, y_true, y_pred, strategy='weighted'):
        return metrics.recall_score(y_true, y_pred, labels=self.labels,
            average=strategy)

    def f1_score(self, y_true, y_pred, strategy='weighted'):
        return metrics.f1_score(y_true, y_pred, labels=self.labels,
            average=strategy)

    def confusion_matrix(self, y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred, labels=self.labels)


class BaselineModel(Model):
    """
    The baseline is a very simple mechanism to categorize any question that
    contains who as who question, when as when questions and what as what
    questions
    """

    def train(self, train, *args, **kwargs):
        pass

    def predict(self, test, model=None):
        y_pred = []
        for question in test:
            if constants.WHEN_TYPE in question:
                y_pred.append(constants.WHEN_TYPE)
            elif constants.WHO_TYPE in question:
                y_pred.append(constants.WHO_TYPE)
            elif constants.WHAT_TYPE in question:
                y_pred.append(constants.WHAT_TYPE)
            else:
                y_pred.append(constants.UNKNOWN_TYPE)
        return y_pred


class NaiveBayesModel(Model):
    """
    The Naive Bayes model uses the Textblob module to build a classifier
    that can predict questions in their respective categories
    """
    def train(self, train, *args, **kwargs):
        from textblob.classifiers import NaiveBayesClassifier

        dataset = []
        for question in train:
            dataset.append((question.text, question.type))
        classifier = NaiveBayesClassifier(dataset)
        return classifier

    def predict(self, test, model):
        y_pred = []
        for question in test:
            category = model.classify(question.text)
            y_pred.append(category)
        return y_pred


class DeepLearningModel(Model):
    """
    The Logistic Regression Model uses scikit-learn's LR model to build a
    classifier based on the Glove Representations of the questions
    """
    def __init__(self, glove_data, dimensions):
        self.glove_data = glove_data
        self.dimensions = dimensions
        self.encoder    = LabelEncoder()
        self.encoder.fit(constants.ALL_TYPES)
        self.logger     = logging.getLogger('models.DeepLearningModel')
        self.dictionary = gensim.models.Word2Vec.load_word2vec_format(
                            self.glove_data, binary=False)

    def train(self, train, *args, **kwargs):
        import numpy as np
        from tensorflow.contrib import learn as skflow

        if 'hidden_units' not in kwargs:
            kwargs['hidden_units'] = [10, 20, 10]
        self.logger.info('Hidden Units = {}'.format(kwargs['hidden_units']))
        if 'n_classes' not in kwargs:
            kwargs['n_classes'] = 5
        self.logger.info('n_classes = {}'.format(kwargs['n_classes']))
        if 'steps' not in kwargs:
            kwargs['steps'] = 5000
        self.logger.info('Number of steps = {}'.format(kwargs['steps']))

        train_labels = train.target()
        self.logger.info('Loading dictionary from {}'.format(self.glove_data))

        self.logger.info('Creating vectors for each question')
        x_train = np.asarray([self.create_vector(question) for question \
                    in train])
        y_train = self.encoder.transform(train_labels)
        self.logger.info('Encoded classes = {}'.format(self.encoder.classes_))

        classifier = skflow.TensorFlowDNNClassifier(**kwargs)

        self.logger.info('Fitting model')
        classifier.fit(x_train, y_train)
        return classifier

    def predict(self, test, model):
        import numpy as np

        x_test = np.asarray([self.create_vector(question) for question \
                    in test])
        self.logger.debug('Predicting from model')
        y_pred = self.encoder.inverse_transform(model.predict(x_test))
        return y_pred

    def create_vector(self, question):
        import numpy as np
        import operator

        vector = np.zeros(self.dimensions)
        count = 2.0
        try:
            if len(question.tokens) == 0:
                return vector
            else:
                vector = map(operator.add,
                             self.dictionary[question.tokens[0].text.lower()],
                             vector)
                if len(question.tokens) == 1:
                    return np.asarray(vector)
                vector = map(operator.add,
                             self.dictionary[question.tokens[1].text.lower()],
                             vector)
                if (question.tokens[0].text.lower() == 'what' and
                        question.tokens[1].text.lower() == 'is'):
                    count = 0.0
                    vector = np.zeros(self.dimensions)
                    for token in question.tokens:
                        count += 1
                        try:
                            vector = map(operator.add,
                                         self.dictionary[token.text.lower()],
                                         vector)
                        except KeyError:
                            count -=1
                    if count == 0:
                        return np.asarray(vector)
                    return np.asarray(vector) / count
                return np.asarray(vector) / count
        except KeyError:
            return vector
