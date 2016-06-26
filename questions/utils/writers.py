from questions.structures.containers import Dataset, Question

class AnnotationWriter(object):
    """
    The dataset is written in the form:

    what    what is the best ebook reader ?
    when    when does the train arrive ?
    ...

    The dataset is written as a raw text file
    """

    DELIMITER = '\t'
    NEW_LINE  = '\n'

    def __init__(self, output_file):
        self.output_file = output_file

    def write(self, dataset):
        with open(self.input_file, 'w') as file_writer:
            for question in dataset:
                output_line = '{0}{1}'.format(AnnotationWriter.DELIMITER.join(
                        [question.type, question.text]),
                        AnnotationWriter.NEW_LINE)
                file_writer.write(output_line)
