from questions.utils.readers import AnnotationReader
from questions.learning.models import DeepLearningModel
from questions.preprocessing.tokenizers import SimpleSpaceTokenizer
from questions.learning.models import Evaluator
from questions.utils.constants import ALL_TYPES

import logging
import numpy as np
import tabulate

class CrossValidation:
    def __init__(self, data_path, reader=None, tokenizer=None):
        self.data_path = data_path
        if not tokenizer:
            self.tokenizer = SimpleSpaceTokenizer()
        else:
            self.tokenizer = tokenizer
        if not reader:
            self.reader = AnnotationReader(self.data_path, self.tokenizer)
        else:
            self.reader = reader
        self.dataset = self.reader.parse()
        self.logger = logging.getLogger('questions.cross_validation')

    def evaluate(self, trainer, evaluator=Evaluator(ALL_TYPES), folds=5):
        split = self.dataset.n_fold_split(n=folds)
        index = 1

        precision = np.zeros(folds+1)
        recall    = np.zeros(folds+1)
        f1_scores = np.zeros(folds+1)

        for training_data, testing_data in split:
            self.logger.info('Training model on fold {}'.format(index))
            model = trainer.train(training_data)
            y_true = testing_data.target()
            y_pred = trainer.predict(testing_data, model)

            # metrics
            pre = evaluator.precision(y_true, y_pred)
            rec = evaluator.recall(y_true, y_pred)
            f1  = evaluator.f1_score(y_true, y_pred)

            precision[index-1] = pre
            recall[index-1]    = rec
            f1_scores[index-1] = f1

            index += 1

        precision[index-1] = np.average(precision[:index-1])
        recall[index-1]    = np.average(recall[:index-1])
        f1_scores[index-1]  = np.average(f1_scores[:index-1])
        return precision, recall, f1_scores

def cross_validation(data_path, glove_path, dimensions, folds=5):
    CV  = CrossValidation(data_path)
    DNN = DeepLearningModel(glove_path, dimensions)

    precision, recall, f1_scores = CV.evaluate(DNN, folds=folds)

    headers = [['Folds', 'Precision', 'Recall', 'F1-Score']]
    folds_columns = ['Fold {}'.format(i+1) for i in range(folds)] + ['Average']
    matrix = np.matrix([folds_columns, precision, recall, f1_scores])

    table = headers + matrix.transpose().tolist()

    print tabulate.tabulate(table, headers='firstrow', floatfmt='.3f')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    data_path  = 'resources/dataset/data.txt'
    glove_path = '/Users/bssubbu/Documents/Projects/data/glove.6B.50d.txt'
    dimensions = 50

    cross_validation(data_path, glove_path, dimensions)
