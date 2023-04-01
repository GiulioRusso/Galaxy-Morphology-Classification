import torch


class ToTensor(object):
    """ Convert Numpy array in tensor keeping the value of the Numpy array"""

    def __call__(self, sample):
        # get the image
        image = sample['image']

        # transform in tensor
        image = torch.from_numpy(image).float()

        # from image (HxWxC) torch.Size([256, 256, 3]) to (CxHxW) torch.Size([3, 256, 256])
        # mean and std dimension (3,) have to match the first dimension of the tensor
        image = image.permute(2, 0, 1)

        # put it into the sample
        sample = {
            'filename': sample['filename'],
            'image': image,
            'label': sample['label']
        }

        return sample
