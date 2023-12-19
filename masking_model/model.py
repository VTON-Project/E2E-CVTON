from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

if TYPE_CHECKING:
    from os import PathLike
    from torch import Tensor

class Masker(nn.Module):
    def __init__(self, lr: float = 0.001, load_from: str | None = None, device: str | None = None):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights='DEFAULT' if load_from is None else None)
        self.model.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.model.aux_classifier = None
        
        self.output_actv = nn.Sigmoid()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if load_from is not None:
            self.load_model(load_from)
            
        if device is not None: self.to(device)
        self.device = device
            
        self.trainmode(False)
    
    def forward(self, X: 'Tensor', training: bool = False):
        X = X.to(self.device)
        
        if X.dim() == 3: X.unsqueeze(0)
        
        if training:
            return self.model(X)
        else:
            with torch.no_grad():
                output = self.output_actv(self.model(X))
                output[output < 0.5] = 0
                output[output >= 0.5] = 1
                
        return output

    def trainmode(self, active: bool):
        self.train(active)
        for p in self.parameters():
            p.requires_grad = active

    def load_model(self, model_path: 'str | PathLike'):
        self.load_state_dict(torch.load(model_path))
        