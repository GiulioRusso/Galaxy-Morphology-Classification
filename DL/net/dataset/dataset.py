import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class GalaxyDataset(Dataset):
    """ Overload of the Dataset class from torch.utils.data """

    def __init__(self, annotations_path, images_path, transforms):
        """
        :param annotations_path: path of the .csv file organized as follow: [FILENAME (the name of the files) | CLASS
        (the class label of the file belong to) | SPLIT (train/validation/test)]
        :param images_path: images folder path
        :param transforms: transform to apply to the dataset
        """

        # read the .csv file as a pandas data frame
        self.images_label = pd.read_csv(annotations_path)
        # get the images folder path
        self.images_path = images_path
        # apply the transforms
        self.transforms = transforms

    def __len__(self):
        """
        :return: length of the dataset using the number of images labels stored into the .csv file
        """

        return len(self.images_label)

    def __getitem__(self, idx):
        """
        :param idx: select a specific item of the dataset using the specified index
        :return: the item accessible like dataset[index][dict{'filename','images','label'}]
        """

        # obtain the file with the name of the image file and it's path using the index 'idx'
        filename = "Galaxy10_DECals-dataset-{}".format(str(idx).zfill(5))
        image_path = os.path.join(self.images_path, filename + '.png')
        image = io.imread(image_path)

        # obtain its label from the images_label dataframe
        label = self.images_label.iloc[idx, 1]

        # all the infos are stored into a dict object organized as 'filename', 'image' and 'label'
        sample = {
            'filename': filename,
            'image': image,
            'label': label
        }

        # apply the transforms to the sample
        if self.transforms:
            sample = self.transforms(sample)

        return sample
