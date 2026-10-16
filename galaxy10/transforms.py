import torch
from torchvision.transforms import v2

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class D4:
    """Random element of the dihedral group D4: k*90 deg rotation, then optional flip.

    Galaxy cutouts have no preferred orientation or handedness, so all 8 are label-preserving.
    """

    def __call__(self, x):
        x = torch.rot90(x, int(torch.randint(4, ())), dims=(-2, -1))
        return x.flip(-1) if torch.rand(()) < 0.5 else x


def build(train, crop=224):
    """uint8 HWC array (256x256) -> normalised float CHW tensor (crop x crop)."""
    head = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    tail = [v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return v2.Compose(head + [D4(), v2.RandomCrop(crop), v2.ColorJitter(brightness=0.1, contrast=0.1)] + tail)
    return v2.Compose(head + [v2.CenterCrop(crop)] + tail)
