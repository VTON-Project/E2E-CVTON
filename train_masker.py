from pathlib import Path

import torch
from torch.utils.data import DataLoader

from masking_model import Masker, MaskerDataset

# Config -------------------------------------------------------------------

CKPT_PATH = Path("masking_model/checkpoints")
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------

train_ds = MaskerDataset("data/viton/data/image", "data/viton/data/mask")
test_ds = MaskerDataset("data/viton/data/image", "data/viton/data/mask") # for now

train_dl = DataLoader(train_ds, BATCH_SIZE, True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_ds, BATCH_SIZE, False, num_workers=4, pin_memory=True)

model = Masker('pretrained', DEVICE)
model.train_model(train_dl, test_dl, CKPT_PATH / "last.pt", CKPT_PATH / "best.pt",
                  run_json_path=CKPT_PATH / "training.json", 
                  learning_rate=LEARNING_RATE, epochs=EPOCHS)