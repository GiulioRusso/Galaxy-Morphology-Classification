from skimage import io
import sys

import pandas as pd
import torch
from torchvision import transforms

from net.dataset.transforms.MinMaxNormalization import MinMaxNormalization
from net.dataset.transforms.StandardNormalization import StandardNormalization
from net.dataset.transforms.ToTensor import ToTensor

def print_transform_std():
    print("Transform applied on image:")
    print(">>> from <class 'numpy.ndarray'> with shape (HxWxC) (256, 256, 3) to (CxHxW) torch.Size([3, 256, 256])")

def dataset_transform(norm, statistics_path):
    """
    :param norm: string with the transformation we want to apply on the dataset
    :param statistics_path: dataset statistics path
    :return: three transforms.Compose object (in order train, validation and test) to apply to an object dataset
    """

    if norm == 'none':

        transforms_train = transforms.Compose([
            ToTensor()
        ])

        transforms_validation = transforms.Compose([
            ToTensor()
        ])

        transforms_test = transforms.Compose([
            ToTensor()
        ])

        return transforms_train, transforms_validation, transforms_test

    elif norm == 'std':

        # read the .csv file
        df = pd.read_csv(statistics_path)

        # extract the mean and std columns
        mean = torch.tensor(df[['MEAN_0', 'MEAN_1', 'MEAN_2']].values.tolist()[0])
        std = torch.tensor(df[['STD_0', 'STD_1', 'STD_2']].values.tolist()[0])

        transforms_train = transforms.Compose([ToTensor(),
                                               StandardNormalization(mean=mean,
                                                                     std=std)])

        transforms_validation = transforms.Compose([ToTensor(),
                                                    StandardNormalization(mean=mean,
                                                                          std=std)])

        transforms_test = transforms.Compose([ToTensor(),
                                              StandardNormalization(mean=mean,
                                                                    std=std)])

        # print_transform_std()

        return transforms_train, transforms_validation, transforms_test

    elif norm == 'min-max':

        # read the .csv file
        df = pd.read_csv(statistics_path)

        # extract the mean and std columns
        min_value = torch.tensor(df[['MIN_0', 'MIN_1', 'MIN_2']].values.tolist()[0])
        max_value = torch.tensor(df[['MAX_0', 'MAX_1', 'MAX_2']].values.tolist()[0])

        transforms_train = transforms.Compose([ToTensor(),
                                               MinMaxNormalization(min_value=min_value,
                                                                   max_value=max_value)])

        transforms_validation = transforms.Compose([ToTensor(),
                                                    MinMaxNormalization(min_value=min_value,
                                                                        max_value=max_value)])

        transforms_test = transforms.Compose([ToTensor(),
                                              MinMaxNormalization(min_value=min_value,
                                                                  max_value=max_value)])

        return transforms_train, transforms_validation, transforms_test

    else:
        print("ERROR: wrong normalization type in .net/dataset/dataset_transform.py function", file=sys.stderr)
