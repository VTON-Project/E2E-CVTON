from torch import nn
import torch.nn.functional as F



class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.first_norm = nn.BatchNorm2d(norm_nc, affine=False)
        ks = 3
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.first_norm(x)

        if tuple(x.shape[2:]) != tuple(segmap.shape[2:]):
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
