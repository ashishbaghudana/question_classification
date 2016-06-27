from questions.utils.readers import CommandLineInterfaceReader
from questions.preprocessing.tokenizers import NLTKTokenizer
from questions.learning.models import DeepLearningModel
from tensorflow.contrib import learn as skflow
from termcolor import cprint
import logging
import numpy as np
import argparse

class Classifier:
    def __init__(self, model_dir, glove_path, dimensions):
        self.model_dir = model_dir
        self.model = skflow.TensorFlowEstimator.restore(self.model_dir)
        self.reader = CommandLineInterfaceReader(NLTKTokenizer())
        self.classifier = DeepLearningModel(glove_path, dimensions)
        self.logger = logging.getLogger('questions.classify')

    def predict(self, text):
        dataset = self.reader.parse(text)
        y_pred = self.classifier.predict(dataset, self.model)
        return y_pred[0]

def classify(model_dir, glove_path, dimensions=50):
    clf = Classifier(model_dir, glove_path, dimensions)
    print "Type exit() to exit"
    print "Enter question: "
    text = raw_input()
    while (text != 'exit()'):
        cprint(clf.predict(text), 'green')
        print "Enter question: "
        text = raw_input()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Classify questions')
    parser.add_argument('-m', '--model', default='resources/model',
            help='Path to dataset')
    parser.add_argument('-g', '--glove', default='resources/glove/glove.6B.50d.txt',
            help='Path to Glove vectors')
    parser.add_argument('-n', '--dimensions', type=int, default=50,
            help='Dimensions')

    args = parser.parse_args()
    classify(args.model, args.glove, args.dimensions)
