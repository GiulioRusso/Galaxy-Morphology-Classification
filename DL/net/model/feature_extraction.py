import csv
import os.path

import torch
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet50_Weights
from net.dataset.dataset_trasform import dataset_transform


def feature_extraction(dataset, dataset_statistics_path):
    """ Extract the features from a ResNet50 network """

    # choose the name of the features file
    file = os.path.join("dataset", "annotations", "cnn-features-merged.csv")

    # apply transform to the dataset
    dataset_copy = dataset
    t, _, _ = dataset_transform(norm='std', statistics_path=dataset_statistics_path)
    dataset_copy.transforms = t

    # get the pre-trained weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # change the last layer into 10 output neurons for out 10 classes problem
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)

    # remove the fully connected layer
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)

    # set the model to evaluation mode
    model.eval()

    # create the CSV file and write rows to it
    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)
        vector = list(range(0, 2048))
        writer.writerow(['FILENAME'] + ['CLASS'] + vector)

    # Load your dataset and pass the images through the network to extract features
    for item in dataset_copy:
        name = item['filename']
        image = item['image']
        target = item['label']
        image = torch.unsqueeze(image, 0)  # add a new dimension for the batch size

        # extract features from the image
        with torch.no_grad():
            features_tensor = model(image)  # torch.Size([1, 2048, 1, 1])

        features_flatten = features_tensor.view(features_tensor.size(0), -1)  # torch.Size([1, 2048])

        # create the row to write in the .csv
        row = []
        row.append(name)
        row.append(target)
        for r in features_flatten:
            for element in r:
                row.append(element.item())

        # open the .csv file in append mode and write the features to the row
        with open(file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
