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
    def __init__(self):
        self.dp_segmenter = DenseposeSegmenter("densepose_utils/densepose_rcnn_R_50_FPN_s1x.yaml")
        self.masker = Masker(Config.masker_path)
        self.image_size = (Config.img_size, Config.img_size * 0.75)

        Config.dp_nc = 25

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])


    def __call__(self, person_img_bgr: 'ndarray',
                 cloth_img_bgr: 'ndarray') -> 'dict[str, Tensor | dict[str, Tensor]]':

        densepose_labels = self.dp_segmenter(person_img_bgr)
        densepose_labels = cv2.resize(densepose_labels, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        densepose_labels = torch.from_numpy(densepose_labels).to(Config.device)
        densepose_seg = torch.zeros(Config.dp_nc + 1, *self.image_size, dtype=torch.float, device=Config.device)
        densepose_seg.scatter_(0, densepose_labels, 1.)
        del densepose_labels

        image = cv2.cvtColor(person_img_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size[::-1], interpolation=cv2.INTER_AREA)

        cloth_image = cv2.cvtColor(cloth_img_bgr, cv2.COLOR_BGR2RGB)

        mask = self.masker.predict_mask(person_img_bgr)
        mask = cv2.resize(mask, self.img_size[::-1], interpolation=cv2.INTER_NEAREST)

        masked_image = image * (1 - mask)

        masked_image = self.transform(masked_image).to(Config.device)
        masked_image = (masked_image - 0.5) / 0.5
        cloth_image = self.transform(cloth_image).to(Config.device)
        cloth_image = (cloth_image - 0.5) / 0.5

        return {"I_m": masked_image, "C_t": cloth_image}, densepose_seg
