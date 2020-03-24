"""
This main class of the exercise performs the data preprocessing, text preprocessing, training and
evaluation of the chosen models.
(As explained in the report) The chosen model for this machine learning task is ordinal regression, to be performed by
binary classification that enables using any requested model. Logistic regression was chosen for the training task.

Training an one-vs-all binary classifier for comparison with the ordinal regression.
"""

import pandas as pd
import numpy as np
import pickle
import data_preperation
import text_preprocessing
import estimate_model
import OrdinalClassifier as Ordinal
from sklearn.linear_model import SGDClassifier

MOVIES_REVIEW_PATH = "movies.txt"
TEST_SET_PATH = "test_set.csv"
TRAIN_SET_PATH = "train_set.csv"
VALIDATION_SET_PATH = "validation_set.csv"
CHUNKSIZE = 10 ** 5
N_ITER = 1
FEATURES_LIST = ['review/text', 'review/score']
TEST_PERCENTAGE = 0.2
VALIDATION_PERCENTAGE = 0.2
RANDOM = 42 # meaning of life


class AmazonReview:

    def __init__(self, movie_reviews_path, features_and_y, n_iter, chunk_size):
        """
        Initializing an object with the my defined models to train for this problem: logistic-sgd + ordinal trained
        with binary classification (with logistic-sgd).
        Saves the classifiers for each model, and hyper-parameters (although in this exercise no hyper-parameters as
        alpha / regularization / norm were defined.
        :param movie_reviews_path: the path to the movie review text file
        :param features_and_y: list of requested features to be used in the model
        :param n_iter: number of iterations
        :param chunk_size: chunk size to iterate the data set each time in various functions
        """
        self.movie_reviews_path = movie_reviews_path
        self.features_and_y_list = features_and_y
        self.n_iter = n_iter
        self.chunk_size = chunk_size
        self.model_names = {1: "ordinal binary classifier",
                            2: "sgd one vs all classifier"}
        self.ordinal_classifier = Ordinal.OrdinalClassifier()
        self.ordinal_classifiers = self.ordinal_classifier.classifiers
        self.sgd_one_vs_all_classifier = SGDClassifier(loss='log')

    def preprocess_data(self):
        """
        Preprocessing data pipeline:
            - turn txt to csv
            - split the csv between training + validation + test sets
            - preprocess the text in each of the sets
        Saves files of processed training, validation and test sets
        :return: void
        """
        # define specific path's
        csv_path = self.movie_reviews_path[:self.movie_reviews_path.find('.')] + ".csv"
        train_validation_path = "train_validation.csv"
        train_processed_path = "processed_" + TRAIN_SET_PATH
        validation_processed_path = "processed_" + VALIDATION_SET_PATH
        test_processed_path = "processed_" + TEST_SET_PATH

        # convert txt to csv
        data_preperation.convert_txt_to_csv(self.movie_reviews_path, csv_path, self.features_and_y_list)

        # split between (train,validation) and (test)
        data_preperation.split_dataset(csv_path, TEST_SET_PATH, train_validation_path, TEST_PERCENTAGE,
                                       CHUNKSIZE, RANDOM)
        # split between (train) and (validation)
        data_preperation.split_dataset(train_validation_path, VALIDATION_SET_PATH, TRAIN_SET_PATH,
                                       ((1-TEST_PERCENTAGE) * VALIDATION_PERCENTAGE), CHUNKSIZE, RANDOM)

        # process the text in each of the sets
        text_preprocessing.text_preprocess(TRAIN_SET_PATH, self.features_and_y_list, train_processed_path)
        text_preprocessing.text_preprocess(VALIDATION_SET_PATH, self.features_and_y_list, validation_processed_path)
        text_preprocessing.text_preprocess(TEST_SET_PATH, self.features_and_y_list, test_processed_path)

    def fit(self, n_iter, chunks, data_path, random):
        """
        Training the models chosen in the exercise (sgd + binary ordinal).
        Saves the values to the object's classifiers
        :param n_iter: number of iterations on all the dataset
        :param chunks: size of chunks to perform the evaluation
        :param data_path: path of data being evaluated (usually training sets)
        :return: void
        """
        for iteration in range(n_iter):
            print("Starting iteration {0}".format(iteration + 1))
            for chunk in pd.read_csv(data_path, chunksize=chunks, names=self.features_and_y_list):
                # randomly alter the training batch
                np.random.seed(random)
                chunk = chunk.iloc[np.random.permutation(len(chunk))]
                X = chunk[self.features_and_y_list[:-1]]
                X = text_preprocessing.feature_extraction_from_text(X, n_features=2**19)
                y = chunk[self.features_and_y_list[-1:]].astype('int32')

                # training the 2 models on the same chunk of hashed vectorizer (harder to compute so being paralleled)
                self.ordinal_classifiers = self.ordinal_classifier.partial_fit(X, y)
                self.sgd_one_vs_all_classifier.partial_fit(X, y['review/score'], classes=np.unique(y))

    def evaluate_models(self, chunks, data_path):
        """
        Evaluating all models in the class (sgd + binary ordinal) performance by chunks
        :param chunks: size of chunks to perform the evaluation
        :param data_path: path of data being evaluated (usually validation / test sets)
        :return: void, but prints the evaluation of the model
        """
        y_ordinal_prediction = np.array([]).astype(int)
        y_sgd_prediction = np.array([]).astype(int)
        y = np.array([]).astype(int)
        for chunk in pd.read_csv(data_path, chunksize=chunks, names=self.features_and_y_list):
            X = chunk[self.features_and_y_list[:-1]]
            X = text_preprocessing.feature_extraction_from_text(X, n_features=2**19)
            y = np.append(y, (chunk[self.features_and_y_list[-1:]].astype('int32')))

            # predicting the 2 models for every chunk and appending to a list to be used afterwards
            y_ordinal_prediction = np.append(y_ordinal_prediction, (self.ordinal_classifier.predict(X)))
            y_sgd_prediction = np.append(y_sgd_prediction, (self.sgd_one_vs_all_classifier.predict(X)))
        self.calculate_and_save_results("ordinal binary classifier", y, y_ordinal_prediction, [1, 2, 3, 4, 5])
        self.calculate_and_save_results("sgd one vs all classifier", y, y_sgd_prediction, [1, 2, 3, 4, 5])

    def pipeline(self, existing_weights):
        """
        Implementing a pipeline for all the process of this exercise:
            - Preprocessing the data
            - Processing the text + feature extraction
            - Training the data on two different models
            - Evaluating the model
        :param existing_weights: if true, do not process or train any data, and load existing data from folder
        :return: void
        """
        if not existing_weights:
            self.preprocess_data()
            self.fit(n_iter=N_ITER, chunks=CHUNKSIZE, data_path=("processed_" + TRAIN_SET_PATH), random=RANDOM)
            self.save_all_models()
        self.load_all_models()
        self.evaluate_models(chunks=CHUNKSIZE, data_path=("processed_" + TEST_SET_PATH))

    def save_all_models(self):
        """
        saves all the trained models in the class
        :return: void
        """
        for model_index in self.model_names:
            self.save_model(model_name=self.model_names[model_index])

    def load_all_models(self):
        """
        loads all requested models to perform fitting on
        :return: void
        """
        for model_index in self.model_names:
            self.load_model(model_name=self.model_names[model_index])

    def save_model(self, model_name, path="pickle_model_"):
        """
        Using pickle, saving weights of a specific model after training to disk
        :param model_name: model name to be saved
        :param path: defined "pickle_model_" for convention
        :return: void
        """
        if model_name == "ordinal binary classifier":
            for classifier_index in range(1, len(self.ordinal_classifiers) + 1):
                pickle_file_path = path + model_name.replace(' ', '_') + str(classifier_index) + ".pkl"
                with open(pickle_file_path, 'wb') as file:
                    pickle.dump(self.ordinal_classifiers[classifier_index], file)
        else:
            pickle_file_path = path + model_name.replace(' ', '_') + ".pkl"
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(self.sgd_one_vs_all_classifier, file)

    def load_model(self, model_name, path="pickle_model_"):
        """
        Using pickle, loading existing model
        :param model_name: the model requested
        :param path: defined "pickle_model_" for convention
        :return: void
        """
        if model_name == "ordinal binary classifier":
            for classifier_index in range(1, len(self.ordinal_classifiers) + 1):
                pickle_file_path = path + model_name.replace(' ', '_') + str(classifier_index) + ".pkl"
                with open(pickle_file_path, 'rb') as file:
                    self.ordinal_classifiers[classifier_index] = pickle.load(file)
        else:
            pickle_file_path = path + model_name.replace(' ', '_') + ".pkl"
            with open(pickle_file_path, 'rb') as file:
                self.sgd_one_vs_all_classifier = pickle.load(file)

    def calculate_and_save_results(self, model_name, y_true, y_predicted, label):
        """
        Calculate and save all the evaluations of the model to files (.txt + .png)
        :param model_name: Specific model to save the results to
        :param y_true: The "ground truth" set
        :param y_predicted: The predictions for the "ground truth" set
        :param label: labels of the classes in the model
        :return: outputs the results, returns void
        """
        accuracy = estimate_model.calc_accuracy(y_true, y_predicted, print_values=False)
        rmse = estimate_model.calc_rmse(y_true, y_predicted)
        plt = estimate_model.calc_confusion_matrix(y_true, y_predicted, label, print_values=False)
        acc_string = "The accuracy of {} is: {}\n".format(model_name, accuracy)
        rmse_string = "The RMSE of {} is: {}\n".format(model_name, rmse)

        text_path = "results_" + model_name.replace(' ', '_') + ".txt"
        plot_path = "confusion_matrix_" + model_name.replace(' ', '_') + ".png"

        with open(text_path, 'w') as file:
            file.write(acc_string)
            file.write(rmse_string)
        plt.savefig(plot_path)


# Implementing the model valuation, based on existing weights (if weights doesn't exist, turn existing_weights=False
if __name__ == "__main__":
    amazon_review = AmazonReview(movie_reviews_path=MOVIES_REVIEW_PATH, features_and_y=FEATURES_LIST,
                                 n_iter=N_ITER, chunk_size=CHUNKSIZE)
    amazon_review.pipeline(existing_weights=True)
