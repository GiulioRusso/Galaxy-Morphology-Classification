import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize


def plot_loss(criterion, epoch_vector, losses, epochs, path):
    """
    :param criterion: type of loss
    :param epoch_vector: epoch vector
    :param losses: loss vector for each epoch
    :param epochs: number of epochs
    :param path: path to save the loss
    :return: the plot of the loss on the path specified
    """

    fig = plt.figure()
    plt.title(f"Loss function: {str(type(criterion).__name__)}", fontsize=18, pad=10)
    plt.grid()
    plt.plot(epoch_vector, losses, 'r')
    plt.xlim(1, epochs)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.savefig(path)
    plt.clf()
    plt.close(fig)


def plot_training_validation_accuracies(epoch_vector, train_accuracies, validation_accuracies, epochs, path):
    """
    :param epoch_vector: epoch vector
    :param train_accuracies: train accuracy vector for each epoch
    :param validation_accuracies: validation accuracy vector for each epoch
    :param epochs: number of epochs
    :param path: path to save the train-validation accuracies
    :return: the plot of the train-validation accuracies on the path specified
    """

    fig = plt.figure()
    plt.title("Train and Validation accuracy", fontsize=18, pad=10)
    plt.grid()
    plt.plot(epoch_vector, train_accuracies, 'b--', label='Training')
    plt.plot(epoch_vector, validation_accuracies, 'b', label='Validation')
    plt.xlim(1, epochs)
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.savefig(path)
    plt.clf()
    plt.close(fig)

def plot_f1(epoch_vector, f1_training, f1_validation, epochs, path):

    fig = plt.figure()
    plt.title("F1", fontsize=18, pad=10)
    plt.grid()
    plt.plot(epoch_vector, f1_training, 'b--', label='Training')
    plt.plot(epoch_vector, f1_validation, 'b', label='Validation')
    plt.xlim(1, epochs)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.xlabel("Epochs")
    plt.ylabel("F1 score")
    plt.savefig(path)
    plt.clf()
    plt.close(fig)


def plot_ROC(path_scores):
    """
    :param path_scores: path of the scores.csv file
    :return: the ROC curve for each class
    """

    # read the .csv with scores
    df = pd.read_csv(path_scores)

    # classes to identify
    classes = np.unique(df['CLASS'])

    # loop each class
    for class_number in classes:

        # select all the predicted rows belonging to a specific class
        df_label = df[df['LABEL'] == class_number]

        # binarize the ground truth of the rows selclass selected
        binarized_labels = label_binarize(df_label['CLASS'], classes=classes)

        # get the number of unique classes (ROC AUC is not defined if the predicted labels are only true or false)
        unique_classes = np.unique(binarized_labels[:, class_number])

        # if we don't have a unique class the ROC AUC is defined
        if len(unique_classes) > 1:

            # calculate false positive rate and true positive rate
            fpr, tpr, thresholds = roc_curve(y_true=binarized_labels[:, class_number], y_score=df_label['SCORE'])

            # calculate the area under the ROC curve
            roc_auc = roc_auc_score(y_true=binarized_labels[:, class_number], y_score=df_label['SCORE'])

            # plot the ROC curve
            fig = plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            ROC_image_name = "ROC-" + str(class_number)
            plt.savefig(os.path.join("results", ROC_image_name + ".png"))
            plt.clf()
            plt.close(fig)

        # if we have a unique class (all true or all false) the ROC AUC is NOT defined
        else:
            print("ERROR: ROC AUC of the class number %d is not defined. ValueError: Only one class present in y_true "
                  % class_number, file=sys.stderr)
