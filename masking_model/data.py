from glob import glob
import os
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from os import PathLike

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor



class MaskerDataset(Dataset):
    def __init__(self, person_images: str | PathLike | list[str | PathLike],
                 masks: str | PathLike | list[str | PathLike],
                 img_size_h_w: tuple[int, int]):

        if isinstance(person_images, (str, PathLike)):
            person_images = str(person_images)
            self.people = glob(os.path.join(person_images, "*"))
        elif isinstance(person_images, list):
            self.people = list(map(lambda x: str(x), person_images))
        else:
            raise TypeError("Wrong type for argument 'person_images'")

        if isinstance(masks, (str, PathLike)):
            masks = str(masks)
            self.people = glob(os.path.join(masks, "*"))
        elif isinstance(masks, list):
            self.masks = list(map(lambda x: str(x), masks))
        else:
            raise TypeError("Wrong type for argument 'masks'")

        self.people.sort()
        self.masks.sort()
        self.image_size = img_size_h_w

        assert len(self.people) == len(self.masks), \
                "person images and mask images count mismatch"

    def __len__(self):
        return len(self.people)


    def __getitem__(self, i):
        person = cv2.imread(self.people[i])
        person = cv2.resize(person, self.image_size[::-1])
        person = preprocess_cv2(person)

        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        mask[mask>0] = 1
        return {"person": person,
                "mask": mask}



def preprocess_cv2(img_bgr: 'ndarray') -> 'Tensor':
    """Takes a cv2 image as input and outputs a Tensor which can be given as input
    to 'deeplabv3_mobilenet_v3_large' model in batches."""

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = torch.as_tensor(img_rgb, dtype=torch.float32)
    img = img.permute(2, 0, 1)
    return preprocess_tensor(img)



def preprocess_tensor(img: 'Tensor') -> 'Tensor':
    """Takes a tensor image of shape (n, c, h, w) or (c, h, w) as input and outputs a
    Tensor which can be given as input to 'deeplabv3_mobilenet_v3_large' model in batches."""

    img = ((img / 255) - 0.5) * 2
    return img



def create_masker_dataset_pair(person_folder: str | PathLike,
                               masks_folder: str | PathLike,
                               img_size_h_w: tuple[int, int],
                               split_ratio: float, *,
                               rng_seed: int | None = None) -> tuple[MaskerDataset, MaskerDataset]:

    person_images = glob(os.path.join(str(person_folder), "*"))
    person_images.sort()
    masks = glob(os.path.join(str(masks_folder), "*"))
    masks.sort()

    assert len(person_images) == len(masks), \
            "person images and mask images count mismatch"

    idxs = np.random.default_rng(rng_seed).permutation(len(person_images))
    person_images = list(map(lambda i: person_images[i], idxs))
    masks = list(map(lambda i: masks[i], idxs))

    split1_len = int(len(person_images) * split_ratio)

    return (MaskerDataset(person_images[:split1_len], masks[:split1_len], img_size_h_w),
            MaskerDataset(person_images[split1_len:], masks[split1_len:], img_size_h_w))
