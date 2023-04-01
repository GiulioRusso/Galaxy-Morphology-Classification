import cv2
import h5py
import csv
import os
import numpy as np

# Use the project path on your machine
project_path = ''

# Select the dataset you want to extract:
dataset_name = 'Galaxy10_DECals'
fileh5_path = ''
images_path = ''

# .csv file parameters
# Name the .csv file
# - All the classes: 'all.csv'
# - Desired class example: 'Name-of-the-label-extracted.csv'
labels_file_name = 'all.csv'
# Select the desired label to extract
# - All the classes: label_to_extract = 'all'
# - Desired class example: label_to_extract = 4 will extract only for the label specified the images and the labels
label_to_extract = 'all'
labels_file_path = ''

# To get the images and labels from file
with h5py.File(fileh5_path, 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])

# Extract the labels in .csv file
labels_file = open(labels_file_path, 'w')
writer = csv.writer(labels_file)
writer.writerow(['FILENAME', 'CLASS', 'DESCRIPTION'])
for i in range(len(labels)):
    if dataset_name == 'Galaxy10_DECals':
        if labels[i] == 0:
            label_descriptor = "Disturbed Galaxies"
        elif labels[i] == 1:
            label_descriptor = "Merging Galaxies"
        elif labels[i] == 2:
            label_descriptor = "Round Smooth Galaxies"
        elif labels[i] == 3:
            label_descriptor = "In-between Round Smooth Galaxies"
        elif labels[i] == 4:
            label_descriptor = "Cigar Shaped Smooth Galaxies"
        elif labels[i] == 5:
            label_descriptor = "Barred Spiral Galaxies"
        elif labels[i] == 6:
            label_descriptor = "Unbarred Tight Spiral Galaxies"
        elif labels[i] == 7:
            label_descriptor = "Unbarred Loose Spiral Galaxies"
        elif labels[i] == 8:
            label_descriptor = "Edge-on Galaxies without Bulge"
        elif labels[i] == 9:
            label_descriptor = "Edge-on Galaxies with Bulge"

    if label_to_extract == 'all':
        row = [dataset_name + "-dataset-{}".format(str(i).zfill(5)), labels[i], label_descriptor]
        writer.writerow(row)
    else:
        if labels[i] == label_to_extract:
            row = [dataset_name + "-dataset-{}".format(str(i).zfill(5)), labels[i], label_descriptor]
            writer.writerow(row)

labels_file.close()

# Extract the images
for i in range(len(images)):
    image_name = dataset_name + "-dataset-{}.png".format(str(i).zfill(5))
    cv2.imwrite(os.path.join(images_path, image_name), images[i])