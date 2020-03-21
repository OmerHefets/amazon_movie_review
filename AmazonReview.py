import OrdinalClassifier as ordinal
import pandas as pd
import numpy as np
import pickle
import training_process
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class AmazonReview:

    def __init__(self, reviews_path, features_and_y):
        self.features_and_y_list = features_and_y
        self.reviews_path = reviews_path
        self.ordinal_classifier = ordinal.OrdinalClassifier()
        self.ordinal_classifiers = self.ordinal_classifier.classifiers
        # self.one_vs_all_classifier = OneVsRestClassifier(SVC())
        # self.mutlti_bayes

    def preprocess(self):
        pass

    def fit(self, n_iter, chunks):
        for i in range(n_iter):
            for chunk in pd.read_csv("training_set_processed.csv", chunksize=chunks, names=self.features_and_y_list):
                np.random.seed(42)
                chunk = chunk.iloc[np.random.permutation(len(chunk))]
                X = chunk[self.features_and_y_list[:-1]]
                y = chunk[self.features_and_y_list[-1:]]
                X = training_process.feature_extraction_from_text(X, n_features=2 ** 19)
                print("Hashed the features")
                self.ordinal_classifiers = self.ordinal_classifier.partial_fit(X, y)
                # y_ova = ....
                # self.one_vs_all_classifier = self.one_vs_all_classifier.fit(X, y_ova)
                print("chunk done")

    def save_models(self):
        for classifier_index in range(1, len(self.ordinal_classifiers) + 1):
            pickle_file_path = "pickel_model" + str(classifier_index) + ".pkl"
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(self.ordinal_classifiers[classifier_index], file)

    def load_models(self, path):
        for classifier_index in range(1, len(self.ordinal_classifiers) + 1):
            pickle_path = path + str(classifier_index) + ".pkl"
            with open(pickle_path, 'rb') as file:
                self.ordinal_classifiers[classifier_index] = pickle.load(file)

    def evaluate_models(self, chunks):
        for chunk in pd.read_csv("validation_set_processed.csv", chunksize=chunks, names=self.features_and_y_list):
            X = chunk[self.features_and_y_list[:-1]]
            y = chunk[self.features_and_y_list[-1:]]
            X = training_process.feature_extraction_from_text(X, n_features=2 ** 19)
            self.ordinal_classifier.predict(X)


if __name__ == "__main__":
    process = AmazonReview('a', ['review/text', 'review/score'])
    #process.fit(1, 10 ** 5)
    #process.save_models()
    process.load_models("pickel_model")
    process.evaluate_models(10 ** 5)
