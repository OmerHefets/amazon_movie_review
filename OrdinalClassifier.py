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

    def partial_fit(self, data, n_iter, chunks, features_and_y_list):
        for classifier_index in range(1, 5):
            print("starting classifier target {0}".format(classifier_index))
            clf = self.classifiers[classifier_index]
            for chunk in pd.read_csv(data, chunksize=chunks, names=features_and_y_list):
                np.random.seed(42)
                chunk = chunk.iloc[np.random.permutation(len(chunk))]
                print(chunk.head)
                X = chunk[features_and_y_list[:-1]]
                y = chunk[features_and_y_list[-1:]]
                X = training_process.feature_extraction_from_text(X, n_features=2 ** 19)
                print(X.shape)
                y = training_process.y_for_ordinal_regression(y, target_value=classifier_index)
                clf.partial_fit(X, y, classes=np.unique(y))
                print("chunk done")
            print("finished classifier target {0}".format(classifier_index))
            self.classifiers[classifier_index] = clf
