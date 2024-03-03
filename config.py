import torch


class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = None
    batch_size = 16
    img_size = 512
    bpgm_path = "./bpgm/checkpoints/bpgm_final_256_26_3_viton.pth"
    model_path = "./checkpoints/C-VTON-VITON-HD"
    masker_path = "./masking_model/checkpoints/best.pt"
