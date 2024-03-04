import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from config import Config
from . import norms



class OASIS_Simple(nn.Module):

    def __init__(self):
        super(OASIS_Simple, self).__init__()

        self.oasis = OASIS_Generator()

        self.bpgm = BPGM(opt)
        self.bpgm.eval()

    def forward(self, I_m, C_t, body_seg, cloth_seg, densepose_seg, agnostic=None):
        if agnostic is not None:
            C_transformed = self.transform_cloth_old(agnostic, C_t)
        else:
            C_transformed = self.transform_cloth(densepose_seg, C_t)

        z = torch.cat((I_m, C_t, C_transformed), dim=1)

        seg_dict = {
            "body": body_seg,
            "cloth": cloth_seg,
            "densepose": densepose_seg
        }

        if len(self.opt.segmentation) == 1:
            seg = seg_dict[self.opt.segmentation[0]]
        else:
            seg = torch.cat([seg_dict[mode] for mode in sorted(seg_dict.keys()) if mode in self.opt.segmentation], axis=1)

        x = self.oasis(seg, z)
        return x

    def transform_cloth(self, seg, C_t):
        if self.bpgm is not None:
            with torch.no_grad():
                # grid, _ = self.bpgm(torch.cat((I_m, seg), dim=1), C_t)
                if self.bpgm.resolution != self.opt.img_size:
                    _seg = F.interpolate(seg, size=self.bpgm.resolution, mode="nearest")
                    _C_t = F.interpolate(C_t, size=self.bpgm.resolution, mode="bilinear", align_corners=False)
                    grid = self.bpgm(_seg, _C_t).permute(0, 3, 1, 2)

                    grid = F.interpolate(grid, size=self.opt.img_size, mode="bilinear", align_corners=False)
                    grid = grid.permute(0, 2, 3, 1)
                else:
                    grid = self.bpgm(seg, C_t)

                C_t = F.grid_sample(C_t, grid, padding_mode='border', align_corners=True)

            return C_t
        else:
            return C_t

    def transform_cloth_old(self, agnostic, C_t):
        if self.bpgm is not None:
            with torch.no_grad():
                # grid, _ = self.bpgm(torch.cat((I_m, seg), dim=1), C_t)
                if self.bpgm.resolution != self.opt.img_size:
                    agnostic = F.interpolate(agnostic, size=self.bpgm.resolution, mode="nearest")
                    _C_t = F.interpolate(C_t, size=self.bpgm.resolution, mode="bilinear", align_corners=False)
                    grid = self.bpgm(agnostic, _C_t).permute(0, 3, 1, 2)

                    grid = F.interpolate(grid, size=self.opt.img_size, mode="bilinear", align_corners=False)
                    grid = grid.permute(0, 2, 3, 1)
                else:
                    grid = self.bpgm(agnostic, C_t)

                C_t = F.grid_sample(C_t, grid, padding_mode='border', align_corners=True)

            return C_t
        else:
            return C_t



class OASIS_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        ch = 32
        if Config.img_size[0] == 64:
            self.channels = [16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif Config.img_size[0] == 256:
            self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif Config.img_size[0] == 512:
            self.channels = [16*ch, 16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        elif Config.img_size[0] == 1024:
            self.channels = [16*ch, 16*ch, 16*ch, 16*ch, 8*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        else:
            raise NotImplementedError

        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1]))

        Z_DIM = 9
        self.fc = nn.Conv2d(Config.dp_nc + Z_DIM, self.channels[0], 3, padding=1)
        self.conv_img = nn.Conv2d(self.channels[-1] + Z_DIM, 3, 3, padding=1)

        self.num_res_blocks = int(math.log(Config.img_size[0], 2)) - 2

    def forward(self, seg, z=None):
        scale = 1 / math.pow(2, self.num_res_blocks-1)
        _z = F.interpolate(z, scale_factor=scale, recompute_scale_factor=False)
        _seg = F.interpolate(seg, scale_factor=scale, recompute_scale_factor=False, mode="nearest")

        x = torch.cat((_z, _seg), dim=1)
        x = self.fc(x)

        for i in range(self.num_res_blocks-1, -1, -1):
            # remember, we go i = n -> 0
            scale = 1 / math.pow(2, i)
            _seg = F.interpolate(seg, scale_factor=scale, mode="nearest", recompute_scale_factor=False)
            _z = F.interpolate(z, scale_factor=scale, recompute_scale_factor=False)

            _cat = torch.cat((_seg, _z), dim=1)
            # x = torch.cat((x, _z), dim=1)

            x = self.body[self.num_res_blocks - 1 - i](x, _cat)
            if i > 0:
                x = self.up(x)

        x = torch.cat((x, z), dim=1)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x




class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        Z_DIM = 9
        spade_conditional_input_dims = Config.dp_nc + Z_DIM

        self.norm_0 = norms.SPADE(fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out
