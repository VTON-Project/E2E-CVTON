from typing import TYPE_CHECKING

import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor

if TYPE_CHECKING:
    from os import PathLike
    from numpy import ndarray

class DenseposeSegmenter:
    def __init__(self, cfg_path: 'PathLike | str'):
        "cfg_path: path to config file of R_50_FPN_s1x denspose model."

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(str(cfg_path))
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        self.model = DefaultPredictor(cfg)
        self.result_extractor = DensePoseResultExtractor()

    def __call__(self, image_bgr: 'ndarray') -> 'ndarray':
        # This method needs more optimization so that it generates one-hot segmentations
        # instead of numerical labels

        with torch.no_grad():
            pred_output = self.model(image_bgr)['instances']
        result = self.result_extractor(pred_output)

        x, y, w, h = result[1][0]
        x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())
        labels = result[0][0].labels.cpu().numpy()
        seg = np.zeros(image_bgr.shape[:2], dtype=labels.dtype)
        seg[y:y+h, x:x+w] = labels
        return seg
