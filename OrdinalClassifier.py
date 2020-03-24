"""
The binary ordinal classifier class.
Implements the partial fit and predict methods, same to the scikit-learn modules convention
"""

import numpy as np
from sklearn.linear_model import SGDClassifier


class OrdinalClassifier:
    def __init__(self):
        """
        Initializes four classifiers for the binary classification in the ordinal regression.
        k-1 classifiers needed for k classes, so four classifiers required in this model.
        """
        self.classifiers = {1: SGDClassifier(loss="log"),
                            2: SGDClassifier(loss="log"),
                            3: SGDClassifier(loss="log"),
                            4: SGDClassifier(loss="log")}

    def partial_fit(self, X, y):
        """
        Performs partial fit in batches. For data 'X' implements a SGD mini-batch for each one of the classifiers
        :param X: the input data
        :param y: the class
        :return: the classifiers after performing a SGD with the data 'X'
        """
        for classifier_index in range(1, 5):
            clf = self.classifiers[classifier_index]
            # y values gets 1 for values bigger than current classifer, and 0 otherwise.
            y['after_classification'] = y.iloc[:, 0].apply(lambda x: 1 if x > classifier_index else 0)
            # do the partial fit on the binary classification
            clf.partial_fit(X, y['after_classification'], classes=np.unique(y['after_classification']))
            self.classifiers[classifier_index] = clf
        return self.classifiers

    def predict(self, X):
        """
        Predicts the probability of class based on the binary classification of all classifiers
        As implemented in the binary ordinal regression.
        :param X: input data
        :return: nparray with the corresponding classes of each input row
        """
        proba = {}
        total_probabilities = np.array([])
        for classifier_index in range(1, 5):
            clf = self.classifiers[classifier_index]
            proba[classifier_index] = clf.predict_proba(X)[:, 1]
        for class_index in range(1, 6):
            if class_index == 1:
                # probability = 1 - probability(bigger than 1)
                total_probabilities = np.vstack(1 - proba[class_index])
            elif 1 < class_index < 5:
                # probability = probabillity(bigger than i) - probability(bigger than i-1)
                total_probabilities = np.column_stack((total_probabilities, (proba[class_index-1]-proba[class_index])))
            elif class_index == 5:
                # probability = probability(bigger than 4)
                total_probabilities = np.column_stack((total_probabilities, (proba[class_index-1])))
        # add one to the results because indexes start at 0, but classes range in (1 - 5)
        results = np.argmax(total_probabilities, axis=1) + 1
        return results
