from typing import TYPE_CHECKING
from glob import glob
import os

import torch
from torch.utils.data import Dataset
import cv2

if TYPE_CHECKING:
    from os import PathLike

class MaskerDataset(Dataset):
    def __init__(self, person_folder: 'str | PathLike', mask_folder: 'str | PathLike'):
        person_folder, mask_folder = str(person_folder), str(mask_folder)
        self.people = glob(os.path.join(person_folder, "*")).sort()
        self.masks = glob(os.path.join(mask_folder, "*")).sort()
        
        assert len(self.people) == len(self.masks), "person images and mask images count mismatch"
    
    def __len__(self):
        return len(self.people)
    
    def __getitem__(self, i):
        person = cv2.imread(self.people[i])
        person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
        person = torch.as_tensor(person, torch.float32)
        person = ((person / 255) - 0.5) * 2
        
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        mask = torch.as_tensor(mask)
        mask[mask>0] = 1
        
        return {"person": person,
                "mask": mask}        