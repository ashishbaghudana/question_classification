# Question Classification
## What, When, Who, Affirmative and Unknown

This module classifies different wh-questions, currently handling What, When, Who and Affirmative (Yes-No) questions. All other questions are classified as unknown. The module creates a Deep Neural Network with 3 hidden layers consisting of 10, 20 and 10 components each. Each question is converted to a 50-dimensional vector using the [Stanford Glove pre-trained vectors](http://nlp.stanford.edu/projects/glove/). These vectors are fed through the DNN and trained over 5000 steps, with a training rates at:

| Precision | Recall | F-Measure |
|----------:|-------:|----------:|
| 0.919     | 0.908  | 0.909     |

**NOTE**: These are precision, recall and F-measure on the training set.

A 5-fold cross validation of the dataset has the following results:

| Folds   |   Precision |   Recall |   F1-Score |
|:--------|------------:|---------:|-----------:|
| Fold 1  |       0.740 |    0.732 |      0.734 |
| Fold 2  |       0.812 |    0.804 |      0.806 |
| Fold 3  |       0.751 |    0.737 |      0.735 |
| Fold 4  |       0.767 |    0.756 |      0.759 |
| Fold 5  |       0.769 |    0.766 |      0.765 |
| Average |       0.768 |    0.759 |      0.760 |

### INSTALLATION

Download the pre-trained Glove data from https://1drv.ms/t/s!ApBGR249NCLchGq8VvyRYbj42U7S and place it within `resources/glove`.

```bash
pip install -r requirements.txt
```

### SAMPLE
```bash
$ python classify.py
Enter question:
What is Brienne\'s sword called?
what
Enter question:
Who is the mother of dragons, Queen of the Andals, the Rhoynar and the First Men?
who
Enter question:
Will Donald Trump be president of the US?
yesno
Enter question:
Which company is Tim Cook the CEO of?
who
Enter question:
What time does the train arrive from Bangalore?
when
Enter question:
Why was Jon Snow contemptuously murdered by the Night\'s Watch?
unknown
Enter question:
When will the clock strive twelve?
when
Enter question:
Why did the chicken cross the road?
unknown
Enter question:
Can the chicken cross the road?
yesno
```

### TRAIN
```bash
$ python train.py -h
usage: train.py [-h] [-d DATASET] [-g GLOVE] [-n DIMENSIONS] [-o OUTPUT]

Train data

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Path to dataset
  -g GLOVE, --glove GLOVE
                        Path to Glove vectors
  -n DIMENSIONS, --dimensions DIMENSIONS
                        Dimensions
  -o OUTPUT, --output OUTPUT
                        Path to output directory
```

### CLASSIFY
```bash
$ python classify.py -h
usage: classify.py [-h] [-m MODEL] [-g GLOVE] [-n DIMENSIONS]

Classify questions

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to dataset
  -g GLOVE, --glove GLOVE
                        Path to Glove vectors
  -n DIMENSIONS, --dimensions DIMENSIONS
                        Dimensions
```
