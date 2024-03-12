from typing import TYPE_CHECKING
import json

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from tqdm import tqdm

from .utils import convert_to_path, AverageCalculator
from .data import preprocess_rgb
from config import Config

if TYPE_CHECKING:
    from os import PathLike
    from numpy import ndarray
    from torch import Tensor
    from torch.utils.data import DataLoader


class Masker(nn.Module):
    def __init__(self, load_weights: str | None = None, device = None):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights="DEFAULT" if load_weights=="PRETRAINED" else None)
        self.model.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.model.aux_classifier = None

        self.output_actv = nn.Sigmoid()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

        if device is None:
            device = "cuda" if Config.gpu_ids[0] != -1 else "cpu"

        if load_weights is not None and load_weights != "PRETRAINED":
            self.load_model(load_weights, device)

        self.to_device(device)
        self.trainmode(False)


    def train_model(self, train_dl: 'DataLoader', test_dl: "DataLoader",
                    last_ckpt: 'str | PathLike', best_ckpt: 'str | PathLike',
                    run_json_path: 'str | PathLike', learning_rate: float | None = None,
                    epochs: int | None = None):

        best_ckpt, last_ckpt, run_json_path = convert_to_path(best_ckpt, last_ckpt, run_json_path)
        best_ckpt.parent.mkdir(parents=True, exist_ok=True)
        last_ckpt.parent.mkdir(parents=True, exist_ok=True)
        run_json_path.parent.mkdir(parents=True, exist_ok=True)

        if run_json_path.exists():
            with open(run_json_path, 'r') as f:
                run = json.load(f)
        else:
            if learning_rate is None or epochs is None:
                raise ValueError("learning_rate and epochs must be set if previous run json does not exist")

            run = {'lr': 0, "epochs": 0, "last_epoch": 0, # epochs start from 1
                   "train_losses": [], "test_losses": []}

        if learning_rate is not None: run["lr"] = learning_rate
        if epochs is not None: run["epochs"] = epochs

        for g in self.optimizer.param_groups:
            g['lr'] = run["lr"]

        if last_ckpt.exists(): self.load_model(last_ckpt, self.device)

        calc = AverageCalculator()
        least_loss = self._get_least_loss(run["test_losses"])
        print(f"Lowest test loss yet: {least_loss}")

        for epoch in range(run["last_epoch"]+1, run["epochs"]+1):
            print(f"\nEpoch {epoch}/{run['epochs']}:")

            self.trainmode(True)
            for data in tqdm(train_dl):
                self.optimizer.zero_grad()
                loss = self.loss_func(self(data["person"]),
                                      data["mask"].to(self.device))
                loss.backward()
                self.optimizer.step()

                calc.update(loss.item(), data["person"].shape[0])

            run["train_losses"].append(calc.avg())
            calc.reset()
            print("Training Loss:", run["train_losses"][-1])

            self.trainmode(False)
            for data in tqdm(test_dl):
                with torch.no_grad():
                    loss = self.loss_func(self(data["person"]),
                                          data["mask"].to(self.device))
                    calc.update(loss.item(), data["person"].shape[0])

            run["test_losses"].append(calc.avg())
            calc.reset()
            print("Testing Loss:", run["test_losses"][-1])

            if least_loss > run["test_losses"][-1]:
                torch.save(self.state_dict(), best_ckpt)
                least_loss = run["test_losses"][-1]
                print("New best model saved!")

            torch.save(self.state_dict(), last_ckpt)

            run["last_epoch"] = epoch
            with open(run_json_path, 'w') as f:
                json.dump(run, f, indent=4)

        print("\nTraining complete!")


    def forward(self, X: 'Tensor') -> 'Tensor':
        """For training"""

        X = X.to(self.device)

        if X.dim() == 3:
            X = X.unsqueeze(0)

        return self.model(X)["out"].squeeze(1)


    def predict_mask(self, img: 'ndarray') -> 'ndarray':
        """Generates mask for an RGB image"""

        img = preprocess_rgb(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            output = self.model(img)["out"].squeeze()
            output = self.output_actv(output)
            output[output < 0.5] = 0
            output[output >= 0.5] = 1
            output = output.to(torch.uint8)

        return output.detach().cpu().numpy()


    def trainmode(self, activate: bool):
        self.train(activate)
        for p in self.parameters():
            p.requires_grad = activate

    def load_model(self, model_path: 'str | PathLike', device):
        self.load_state_dict(torch.load(model_path, device))

    def to_device(self, device):
        self.device = device
        return self.to(device)

    @staticmethod
    def _get_least_loss(losses: list):
        if len(losses) == 0: return 100.
        else: return min(losses)
