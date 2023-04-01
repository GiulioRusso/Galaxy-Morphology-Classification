import csv
import os.path
from statistics import mean

import torch


def save_experiment_parameters(path, file_name, model_name, epochs, batch_size, optimizer, scheduler, criterion,
                               learning_rate, lr_step_size, lr_gamma, momentum):
    """ Save all the experiments parameters in a .csv file in the experiment path """

    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(['FILE NAME', 'MODEL_NAME', 'EPOCHS', 'BATCH_SIZE', 'OPTIMIZER', 'SCHEDULER', 'CRITERION',
                         'LEARNING_RATE', 'LR_STEP_SIZE', 'LR_GAMMA', 'MOMENTUM'])
        writer.writerow([file_name, model_name, epochs, batch_size, type(optimizer).__name__, type(scheduler).__name__,
                         type(criterion).__name__, learning_rate, lr_step_size, lr_gamma, momentum])


def save_evaluation_training(path, accuracy, precision, recall, f1):
    """ Print on a new .csv the values of accuracy, precision, recall and f1 """

    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(['TOT ACCURACY (%)', 'AVG_PRECISION', 'AVG_RECALL', 'AVG_F1'])
        writer.writerow([round(accuracy.item(), 2), round(precision, 2), round(recall, 2), round(f1, 2)])


def save_evaluation(path, accuracy_value, accuracy, precision_value, recall_value, f1_value, precision, recall, f1):
    """ Print on the .csv of the experiment parameters accuracy, precision, recall and f1 """

    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(['CLASS', 'ACCURACY (%)', 'PRECISION', 'RECALL', 'F1'])
        for i in range(len(precision)):
            writer.writerow([i,
                             round(accuracy[i].item(), 2),
                             round(precision[i], 2),
                             round(recall[i], 2),
                             round(f1[i], 2)])

        writer.writerow(['TOT / WEIGHTED MEAN',
                         torch.round(accuracy_value).item(),
                         round(precision_value, 2),
                         round(recall_value, 2),
                         round(f1_value, 2)])


def save_batch_training(path, epoch_number, batch_number, names, targets, outputs_max, outputs, precision, recall, f1,
                        correct):
    """ Save batch info for debug in the path specified """

    separator = '|'
    debug_file_name = separator.join(["e=" + str(epoch_number), "b=" + str(batch_number)]) + ".csv"
    batch_length = len(names)

    with open(os.path.join(path, debug_file_name), "w") as file:
        writer = csv.writer(file)
        writer.writerow(['PRECISION', 'RECALL', 'F1', 'CORRECT'])
        writer.writerow([round(precision, 2), round(recall, 2), round(f1, 2), int(correct.item())])

        writer.writerow(['FILENAME', 'CLASS', 'LABEL', 'SCORE'])

        for index in range(0, batch_length):
            writer.writerow([names[index], int(targets[index]), int(outputs_max[index]),
                             float("{:.4f}".format(outputs[index, outputs_max[index]]))])


def save_errors(path, predictions, targets, dataset):
    """
    :param path: path to save the .csv containing the errors
    :param predictions: predictions made by the test function
    :param targets: ground truth
    :param dataset: dataset of test
    :return: save a .csv file in the path specified
    """

    # predictions / target comparisons = 1 for match, 0 for mismatch
    # we invert with ~, so we have 0 for match, 1 for mismatch
    # nonzero elements are thus all mismatches
    errors = torch.nonzero(~predictions.eq(targets))

    # save errors in results/errors.csv with filename, correct class and label predicted
    with open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(['FILENAME', 'CLASS', 'LABEL'])

        for index in errors:
            writer.writerow([dataset[index]['filename'], int(targets[index]), int(predictions[index])])
