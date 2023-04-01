import sys

import pandas as pd
import torch
from pandas import read_csv
from torch.utils.data import Subset
from skimage import io


def print_split(annotations_path, index_train, index_validation, index_test):
    """
    :param annotations_path: path of the .csv to read
    :param index_train: index of train images
    :param index_validation: index of validation images
    :param index_test: index of test images
    :return: print of the division of the dataset
    """

    df = pd.read_csv(annotations_path)

    print("----------------------------------")
    print("Dataset shape: %s elements     |" % str(df.shape[0]))
    print("----------------------------------")
    print("Training set length: %s         |" % str(len(index_train)))
    print("Validation set length: %s       |" % str(len(index_validation)))
    print("Test set length: %s             |" % str(len(index_test)))
    print("----------------------------------")
    print("Class | Train | Validation | Test |")
    print("----------------------------------")

    if 'all-10.csv' in annotations_path:
        print("  0   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'validation')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'test')].shape[0])))
        print("  1   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'validation')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'test')].shape[0])))
        print("  2   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'test')].shape[0])))
        print("  3   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'test')].shape[0])))
        print("  4   |  %d  |    %d      | %d   |" % ((df.loc[(df['CLASS'] == 4) & (df['SPLIT'] == 'train')].shape[0]),
                                                      (df.loc[(df['CLASS'] == 4) & (df['SPLIT'] == 'validation')].shape[0]),
                                                      (df.loc[(df['CLASS'] == 4) & (df['SPLIT'] == 'test')].shape[0])))
        print("  5   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 5) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 5) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 5) & (df['SPLIT'] == 'test')].shape[0])))
        print("  6   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 6) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 6) & (df['SPLIT'] == 'validation')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 6) & (df['SPLIT'] == 'test')].shape[0])))
        print("  7   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 7) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 7) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 7) & (df['SPLIT'] == 'test')].shape[0])))
        print("  8   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 8) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 8) & (df['SPLIT'] == 'validation')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 8) & (df['SPLIT'] == 'test')].shape[0])))
        print("  9   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 9) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 9) & (df['SPLIT'] == 'validation')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 9) & (df['SPLIT'] == 'test')].shape[0])))
        print("----------------------------------")

    elif 'all-4.csv' in annotations_path:
        print("  0   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'test')].shape[0])))
        print("  1   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'test')].shape[0])))
        print("  2   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                   df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'test')].shape[0])))
        print("  3   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                   df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'test')].shape[0])))
        print("----------------------------------")

    elif 'all-3.csv' in annotations_path:
        print("  0   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'test')].shape[0])))
        print("  1   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'test')].shape[0])))
        print("  2   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                   df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'test')].shape[0])))
        print("----------------------------------")

    elif 'all-10-augmented.csv' in annotations_path:
        print("  0   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'test')].shape[0])))
        print("  1   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'test')].shape[0])))
        print("  2   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                   df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'test')].shape[0])))
        print("  3   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                   df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'test')].shape[0])))
        print("  4   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 4) & (df['SPLIT'] == 'train')].shape[0]),
                                                      (df.loc[(df['CLASS'] == 4) & (df['SPLIT'] == 'validation')].shape[
                                                          0]),
                                                      (df.loc[(df['CLASS'] == 4) & (df['SPLIT'] == 'test')].shape[0])))
        print("  5   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 5) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                   df.loc[(df['CLASS'] == 5) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 5) & (df['SPLIT'] == 'test')].shape[0])))
        print("  6   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 6) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 6) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 6) & (df['SPLIT'] == 'test')].shape[0])))
        print("  7   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 7) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                   df.loc[(df['CLASS'] == 7) & (df['SPLIT'] == 'validation')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 7) & (df['SPLIT'] == 'test')].shape[0])))
        print("  8   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 8) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 8) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 8) & (df['SPLIT'] == 'test')].shape[0])))
        print("  9   |  %d  |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 9) & (df['SPLIT'] == 'train')].shape[0]),
                                                    (df.loc[(df['CLASS'] == 9) & (df['SPLIT'] == 'validation')].shape[
                                                        0]),
                                                    (df.loc[(df['CLASS'] == 9) & (df['SPLIT'] == 'test')].shape[0])))
        print("----------------------------------")

    elif 'all-4-augmented.csv' in annotations_path:
        print("  0   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'validation')].shape[
                                                       0]),
                                                   (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'test')].shape[0])))
        print("  1   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'train')].shape[0]),
                                                 (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'validation')].shape[
                                                     0]),
                                                 (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'test')].shape[0])))
        print("  2   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'train')].shape[0]),
                                                 (
                                                     df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'validation')].shape[
                                                         0]),
                                                 (df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'test')].shape[0])))
        print("  3   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                       df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'validation')].shape[
                                                           0]),
                                                   (df.loc[(df['CLASS'] == 3) & (df['SPLIT'] == 'test')].shape[0])))
        print("----------------------------------")

    elif 'all-3-augmented.csv' in annotations_path:
        print("  0   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'train')].shape[0]),
                                                 (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'validation')].shape[
                                                     0]),
                                                 (df.loc[(df['CLASS'] == 0) & (df['SPLIT'] == 'test')].shape[0])))
        print("  1   |  %d |    %d    | %d |" % ((df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'train')].shape[0]),
                                                 (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'validation')].shape[
                                                     0]),
                                                 (df.loc[(df['CLASS'] == 1) & (df['SPLIT'] == 'test')].shape[0])))
        print("  2   |  %d |    %d     | %d  |" % ((df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'train')].shape[0]),
                                                   (
                                                       df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'validation')].shape[
                                                           0]),
                                                   (df.loc[(df['CLASS'] == 2) & (df['SPLIT'] == 'test')].shape[0])))
        print("----------------------------------")



def print_getitem(dataset, img_number, show):
    """
    :param dataset: subset object of which we want to check the get_item method
    :param img_number: number of the object to check
    :param show: flag to show or not the image choosen
    :return: print of the structure of one dataset item
    """

    filename = dataset[img_number]['filename']
    image = dataset[img_number]['image']
    label = dataset[img_number]['label']
    print("\nSample dict")
    print("-------------------------------------------------------------------")
    print("  Dict   |           type          |            value              |")
    print("-------------------------------------------------------------------")
    print("filename |      %s      | %s |" % (str(type(filename)), str(filename)))
    print("  image  | %s |  shape (HxWxC) %s  |" % (str(type(image)), str(image.shape)))
    print("  label  |  %s  |               %s               |" % (str(type(label)), str(label)))
    print("-------------------------------------------------------------------")

    if show:
        # convert image from tensor to numpy to show it
        if torch.is_tensor(image):
            image = image.permute((1, 2, 0))  # permute 3xHxW to HxWx3
            image = image.cpu().detach().numpy()  # conversion only possible with CPU
        io.imshow(image)
        io.show()


def dataset_split(dataset, annotations_path):
    """
    :param dataset: dataset instance of the Dataset class
    :param annotations_path: path to the .csv file that has to be organized in columns as follow [FILENAME (the name of
    the files) | CLASS (the class label of the file belong to) | SPLIT (train/validation/test)]
    :return: subset object of the training, validation and test set
    """

    # read csv data split
    data_split = read_csv(filepath_or_buffer=annotations_path, usecols=["FILENAME", "CLASS", "SPLIT"]).values

    # select the SPLIT columns of the .csv
    split = data_split[:, 2]
    # get the length of the data
    num_data = len(split)

    # init list
    index_train = []
    index_validation = []
    index_test = []

    # get the indexes of the train, validation and test set as marked in the .csv file
    for index in range(num_data):
        if split[index] == 'train':
            index_train.append(index)
        elif split[index] == 'validation':
            index_validation.append(index)
        elif split[index] == 'test':
            index_test.append(index)

    # divide the dataset into the three subset train, validation and test according to the index
    dataset_train = Subset(dataset=dataset,
                           indices=index_train)

    dataset_validation = Subset(dataset=dataset,
                                indices=index_validation)

    dataset_test = Subset(dataset=dataset,
                          indices=index_test)

    # check the split
    print_split(annotations_path=annotations_path,
                index_train=index_train,
                index_validation=index_validation,
                index_test=index_test)

    # check get item
    # print_getitem(dataset=dataset_train, img_number=0, show=False)

    return dataset_train, dataset_validation, dataset_test
