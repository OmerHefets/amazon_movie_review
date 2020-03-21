import pandas as pd
import numpy as np
import training_process
from sklearn.linear_model import SGDClassifier


class OrdinalClassifier:

    def __init__(self):
        self.classifiers = {1: SGDClassifier(loss="log"),
                            2: SGDClassifier(loss="log"),
                            3: SGDClassifier(loss="log"),
                            4: SGDClassifier(loss="log")}

    def partial_fit(self, X, y):
        for classifier_index in range(1, 5):
            print("starting classifier target {0}".format(classifier_index))
            clf = self.classifiers[classifier_index]
            y['after_classification'] = y.iloc[:, 0].apply(lambda x: 1 if x > classifier_index else 0)
            # y = training_process.y_for_ordinal_regression(y, target_value=classifier_index)
            clf.partial_fit(X, y['after_classification'], classes=np.unique(y['after_classification']))
            print("finished classifier target {0}".format(classifier_index))
            self.classifiers[classifier_index] = clf
        return self.classifiers

    def predict(self, X):
        probabilities = {}
        total_probabilities = np.array([])
        for classifier_index in range(1, 5):
            clf = self.classifiers[classifier_index]
            probabilities[classifier_index] = clf.predict_proba(X)[:, 1]
        for class_index in range(1, 6):
            if class_index == 1:
                total_probabilities = np.vstack(1 - probabilities[class_index])
            elif 1 < class_index < 5:
                total_probabilities = np.column_stack((total_probabilities, (probabilities[class_index - 1] - probabilities[class_index])))
            elif class_index == 5:
                total_probabilities = np.column_stack((total_probabilities, (probabilities[class_index - 1])))
        results = list(np.argmax(total_probabilities, axis=1))
        return [x+1 for x in results]
