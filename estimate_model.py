import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def display_confusion_matrix(y_true, y_predicted, labels):
    cm = confusion_matrix(y_true, y_predicted, labels=labels)
    display = ConfusionMatrixDisplay(cm, display_labels=labels)
    display = display.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    plt.show()

