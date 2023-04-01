import warnings
import torch
from sklearn.metrics import precision_recall_fscore_support


def accuracy(predictions, targets, print_flag):
    """
    :param predictions: predictions made by the test function
    :param targets: ground truth
    :param print_flag: flag to print the accuracy values
    :return: accuracy overall value and accuracy for each class
    """

    # calculate overall accuracy
    accuracy_value = 100. * torch.mean((targets == predictions).float())

    # flatten the tensor
    targets_flat = targets.flatten()

    # get the unique values and their counts
    unique_values, value_counts = torch.unique(targets_flat, return_counts=True)

    # calculate accuracy for each class
    num_classes = len(unique_values)

    accuracy_vector_per_class = torch.zeros(num_classes)
    for i in range(num_classes):
        idx = (targets == i)
        accuracy_vector_per_class[i] = 100. * torch.mean((targets[idx] == predictions[idx]).float())

    if print_flag:
        print("\nAccuracy for each class:")
        for i in range(targets.max() + 1):
            print(f"Class {i}: {accuracy_vector_per_class[i]:.2f} %")

        print(f"\nAccuracy on the entire dataset: {accuracy_value:.2f} %")

    return accuracy_value, accuracy_vector_per_class


def precision_recall_f1(y_true, y_pred, print_flag):
    """
    :param y_true: ground truth
    :param y_pred: predictions
    :param print_flag: bool to print the values or not
    :return: the precision, recall and f1 score average and per class
    """

    # ignore the warning message if precision/recall/f1 is zero
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # evaluate metrics per class
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true.cpu().detach().numpy(),
                                                                                             y_pred.cpu().detach().numpy(),
                                                                                             average=None)

    # evaluate metrics weighted
    precision_value, recall_value, f1_value, _ = precision_recall_fscore_support(y_true.cpu().detach().numpy(),
                                                                                 y_pred.cpu().detach().numpy(),
                                                                                 average='weighted')

    if print_flag:
        print("\nPrecision - Recall - F1 for each class:")
        for i in range(len(precision_per_class)):
            print(
                f"Class {i}: Precision {round(precision_per_class[i], 2)} / "
                f"Recall {round(recall_per_class[i], 2)} / "
                f"F1 {round(f1_per_class[i], 2)} ")

        print(f"\nWeighted mean: Precision {round(precision_value, 2)} / "
              f"Recall {round(recall_value, 2)} / "
              f"F1 {round(f1_value, 2)}")

    return precision_value, recall_value, f1_value, precision_per_class, recall_per_class, f1_per_class
