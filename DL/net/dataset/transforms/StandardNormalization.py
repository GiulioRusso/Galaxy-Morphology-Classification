from torchvision.transforms import transforms


class StandardNormalization(object):
    """
    Standard Normalization is a scaling that shift the data by their mean, and bring the standard deviation to 1

                 x - mean
    x_scaled = -------------
                   std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # read sample
        image = sample['image']

        # standard normalization
        transform = transforms.Normalize([self.mean[0], self.mean[1], self.mean[2]],
                                         [self.std[0], self.std[1], self.std[2]])

        # transform the image
        image_normalized = transform(image)

        # put it into the sample
        sample = {
            'filename': sample['filename'],
            'image': image_normalized,
            'label': sample['label']
        }

        return sample
