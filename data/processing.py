from typing import TYPE_CHECKING

import cv2
import torch
from torchvision import transforms

from masking_model import Masker
from config import Config
from densepose_utils.segmenter import DenseposeSegmenter

if TYPE_CHECKING:
    from torch import Tensor
    from numpy import ndarray


class Preprocessor:
    """Takes RGB images as input and outputs preprocessed data to input into VTON model."""

    def __init__(self):
        self.dp_segmenter = DenseposeSegmenter("densepose_utils/densepose_rcnn_R_50_FPN_s1x.yaml")
        self.masker = Masker(Config.masker_path)

        if isinstance(Config.img_size, int):
            Config.img_size = (Config.img_size, int(Config.img_size * 0.75))

        DENSEPOSE_NUM_CHANNELS = 25
        Config.dp_nc = DENSEPOSE_NUM_CHANNELS + 1  # I don't know why, but was present in original implementation

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(Config.img_size),
            transforms.ToTensor(),
        ])

        self.device = "cuda" if Config.gpu_ids[0] != -1 else "cpu"


    def __call__(self, person_img: 'ndarray',
                 cloth_img: 'ndarray') -> 'tuple[dict[str, Tensor], Tensor]':

        person_img_bgr = cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR)
        densepose_labels = self.dp_segmenter(person_img_bgr)
        densepose_labels = cv2.resize(densepose_labels, Config.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        densepose_labels = torch.from_numpy(densepose_labels).to(self.device, torch.int64)
        densepose_seg = torch.zeros(Config.dp_nc, *Config.img_size, dtype=torch.float, device=self.device)
        densepose_seg.scatter_(0, densepose_labels.unsqueeze(0), 1.)
        del densepose_labels

        mask = self.masker.predict_mask(person_img)

        person_img = cv2.resize(person_img, Config.img_size[::-1], interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, Config.img_size[::-1], interpolation=cv2.INTER_NEAREST)[..., None]

        masked_image = person_img * (1 - mask)

        masked_image = self.transform(masked_image).to(self.device)
        masked_image = (masked_image - 0.5) / 0.5
        cloth_img = self.transform(cloth_img).to(self.device)
        cloth_img = (cloth_img - 0.5) / 0.5

        return {"I_m": masked_image, "C_t": cloth_img}, densepose_seg
