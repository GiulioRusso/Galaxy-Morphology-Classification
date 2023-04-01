import csv
import os
import numpy as np


def dataset_statistics(dataset):
    """
    :param dataset: the dataset on which we evaluate statistics value
    :return: the file results/statistics.csv with mean, std, min, max for each channel
    """

    # stack all the images tensors in one tensor (dataset_length, channels, width, height)
    images = np.stack([dataset[index]['image'] for index in range(len(dataset))], axis=0)
    # mean over the stacked tensor for each channel
    mean = np.mean(images, axis=(0, 1, 2))
    # std over the stacked tensor for each channel
    std = np.std(images, axis=(0, 1, 2))
    # min value in the tensor
    min_value = np.min(images, axis=(0, 1, 2))
    # max value in the tensor
    max_value = np.max(images, axis=(0, 1, 2))

    # write the statistics data into a .csv
    with open(os.path.join("results", "statistics.csv"), "w") as file:
        writer = csv.writer(file)
        writer.writerow(['MEAN_0', 'MEAN_1', 'MEAN_2',
                         'STD_0', 'STD_1', 'STD_2',
                         'MIN_0', 'MIN_1', 'MIN_2',
                         'MAX_0', 'MAX_1', 'MAX_2',
                         ])
        writer.writerow([mean[0], mean[1], mean[2],
                         std[0], std[1], std[2],
                         min_value[0], min_value[1], min_value[2],
                         max_value[0], max_value[1], max_value[2]])
