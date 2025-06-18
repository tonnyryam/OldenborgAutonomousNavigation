import albumentations as A
import numpy as np
from fastai.vision.core import PILImage
from fastai.vision.augment import RandTransform


class AlbumentationsTransform(RandTransform):
    """Class that handles albumentations transformations during training."""

    def __init__(self, train_aug, valid_aug=None, split_idx=None):
        """Constructor for AlbumentationsTransform."""
        super().__init__()  # calls base class (RandTransform) constructor
        self.train_aug = train_aug
        self.valid_aug = (
            valid_aug or train_aug
        )  # defaults to training augmentations if no validation augmentations are provided
        self.split_idx = split_idx  # indicates whether the transform is applied to training or validation data
        self.order = 2  # apply after resizing

    def before_call(self, split_idx):
        """Called before the transform is applied to set the split index so we know if it's training or validation."""
        self.idx = split_idx

    def encodes(self, img: PILImage):
        """Apply the Albumentations transformations to the input image."""
        aug = (
            self.train_aug if self.idx == 0 else self.valid_aug
        )  # apply the appropriate augmentation
        image = np.array(img)  # albumentations works with numpy arrays
        image = aug(image=image)[
            "image"
        ]  # extract the image from the augmentation result
        return PILImage.create(image)  # convert back to PILImage for compatibility


def get_train_aug():
    """Data augmentations applied to training data."""
    return A.Compose(
        [
            A.VerticalFlip(p=0.5),  # 50% chance to flip vertically
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5
            ),  # 50% chance to shift, scale, and rotate
            A.RandomBrightnessContrast(
                p=0.2
            ),  # 20% chance to adjust brightness and contrast
        ]
    )


def get_valid_aug():
    """Data augmentations applied to validation data (none)."""
    return A.Compose([])
