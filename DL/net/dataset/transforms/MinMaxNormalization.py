from torchvision import transforms


class MinMaxNormalization(object):
    """
    Min-Max Normalization (usually called Feature Scaling) performs a linear transformation on the original data.
    This technique gets all the scaled data in the range (0, 1)

                       (x - x_min)
        x_scaled  =  ---------------                    for each channel
                     (x_max - x_min)

    min-max normalization preserves the relationships among the original data values
    """

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sample):
        # read sample
        image = sample['image']

        # min max normalization (image range to [0, 1])
        transform = transforms.Normalize([self.min_value[0], self.min_value[1], self.min_value[2]],
                                         [self.max_value[0] - self.min_value[0], self.max_value[1] - self.min_value[1], self.max_value[2] - self.min_value[2]])

        # apply the normalization
        image_normalized = transform(image)

        # put it into the sample
        sample = {
            'filename': sample['filename'],
            'image': image_normalized,
            'label': sample['label']
        }

        return sample
