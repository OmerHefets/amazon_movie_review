import OrdinalClassifier as ordinal
import pandas as pd
import numpy as np
import training_process
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class AmazonReview:

    def __init__(self, reviews_path, features_and_y):
        self.features_and_y_list = features_and_y
        self.reviews_path = reviews_path
        self.ordinal_classifiers = ordinal.OrdinalClassifier()
        self.one_vs_all_classifier = OneVsRestClassifier(SVC())
        # self.mutlti_bayes

    def preprocess(self):
        pass

    def fit(self, n_iter, chunks):
        for i in range(n_iter):
            for chunk in pd.read_csv(data_after_preprocess, chunksize=chunks, names=self.features_and_y_list):
                np.random.seed(42)
                chunk = chunk.iloc[np.random.permutation(len(chunk))]
                print(chunk.head)
                X = chunk[self.features_and_y_list[:-1]]
                y = chunk[self.features_and_y_list[-1:]]
                X = training_process.feature_extraction_from_text(X, n_features=2 ** 19)

    def evaluate_models(self):
        pass

