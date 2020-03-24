"""
This module contains the functions required to evaluate the model's performance:
* Confusion Matrix calculation
* Accuracy calculation
* RMSE calculation
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, mean_squared_error


def calc_confusion_matrix(y_true, y_predicted, labels, print_values):
    """
    Calculates and plots the confusion matrix
    :param y_true: The "ground truth" set
    :param y_predicted: The predictions for the "ground truth" set
    :param labels: labels of the classes in the model
    :param print_values: if true print the confusion matrix instead of return it (will not save it to the computer)
    :return: the plot
    """
    cm = confusion_matrix(y_true, y_predicted, labels=labels)
    display = ConfusionMatrixDisplay(cm, display_labels=labels)
    display = display.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    if print_values:
        plt.show()
    return plt


def calc_accuracy(y_true, y_predicted, print_values):
    """
    Calculating the accuracy of the model (how many correct predictions were made)
    :param y_true: The "ground truth" set
    :param y_predicted: The predictions for the "ground truth" set
    :param print_values: if true will print the accuracy as well
    :return: the accuracy of the model
    """
    accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted, normalize=True)
    if print_values:
        print("The accuracy of the model is: {0}".format(accuracy))
    return accuracy


def calc_rmse(y_true, y_predicted):
    """
    Calculates the RMSE of the predictions, using scikit-learn MSE module
    :param y_true: The "ground truth" set
    :param y_predicted: The predictions for the "ground truth" set
    :return: the RMSE
    """
    return mean_squared_error(y_true=y_true, y_pred=y_predicted) ** 0.5
