import OrdinalClassifier as ordinal
import pandas as pd
import numpy as np
import pickle
import training_process
import estimate_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
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
        total_examples = 0
        correct_ordinal = 0
        for chunk in pd.read_csv("validation_set_processed.csv", chunksize=chunks, names=self.features_and_y_list):
            total_examples += len(chunk)
            X = chunk[self.features_and_y_list[:-1]]
            y = chunk[self.features_and_y_list[-1:]].astype('int32')
            X = training_process.feature_extraction_from_text(X, n_features=2 ** 19)
            y_pred = self.ordinal_classifier.predict(X)
            correct_ordinal += accuracy_score(y, y_pred, normalize=False)
            # estimate_model.display_confusion_matrix(y, y_pred, labels=[1, 2, 3, 4, 5])
        acc = correct_ordinal / total_examples
        print("the accuracy is: {0}".format(acc))


if __name__ == "__main__":
    process = AmazonReview('a', ['review/text', 'review/score'])
    #process.fit(1, 10 ** 5)
    #process.save_models()
    process.load_models("pickel_model")
    process.evaluate_models(10 ** 5)
