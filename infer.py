from typing import TYPE_CHECKING

import cv2
import numpy as np

from data.processing import Preprocessor
from models import OASIS_model
from models.utils import put_on_multi_gpus
from config import Config

if TYPE_CHECKING:
    from numpy import ndarray



class VTONInference:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = OASIS_model('test')
        self.model.eval()

        if Config.gpu_ids[0] != -1: model = put_on_multi_gpus(model)


    def __call__(self, person_img_bgr: 'ndarray', cloth_img_bgr: 'ndarray') -> 'ndarray':
        image, seg = self.preprocessor(person_img_bgr, cloth_img_bgr)

        for k, v in image.items():
            image[k] = v.unsqueeze(0)
        seg = seg.unsqueeze(0)

        pred = self.model(image, seg, "generate", None).detach().cpu().squeeze().permute(1, 2, 0).numpy()

        pred = (pred + 1) / 2
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        pred = cv2.resize(pred, Config.img_size[::-1], interpolation=cv2.INTER_AREA)

        return pred
