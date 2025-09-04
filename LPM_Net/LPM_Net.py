
import torch.nn.functional as F


import os
os.environ['CUDA_VISIBLE_DEVICE']='0'
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from FANLayer import FANLayer
from functools import partial



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        return self.conv(input)


# Conv_Block
class Convblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Convblock, self).__init__()
        self.encoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.ebn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.ebn(self.encoder(x)), 2, 2))
        return out


# Up_Block
class FAN_Upblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FAN_Upblock, self).__init__()
        self.decoder = FANConv(input_channels, output_channels)
        self.dbn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        out = F.relu(F.interpolate(self.dbn(self.decoder(x)), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        return out

'''------------------------------------PIMM------------------------------------------------'''



class PIMM(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=1, kernel_size=3, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = FANConv(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.split_indices1 = ((hidden - conv_channels) // 2, (hidden - conv_channels) - (hidden - conv_channels) // 2)
        self.split_indices2 = (conv_channels // 2, conv_channels - conv_channels // 2)


        self.fc5 = FANLayer((hidden - conv_channels) // 2 + conv_channels // 2, (hidden - conv_channels) // 2 + conv_channels // 2)
        self.fc6 = FANLayer(((hidden - conv_channels) - (hidden - conv_channels) // 2) + conv_channels - conv_channels // 2, ((hidden - conv_channels) - (hidden - conv_channels) // 2) + conv_channels - conv_channels // 2)

        self.fc7 = FANLayer(hidden, dim)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        i1, i2 = torch.split(i, self.split_indices1, dim=-1)  # [B, H, W, C]


        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        c1, c2 = torch.split(c, self.split_indices2, dim=-1)  # [B, H, W, C]

        out1 = torch.cat((i1, c1), dim=-1)
        out1 = self.fc5(out1)
        out2 = torch.cat((i2, c2), dim=-1)
        out2 = self.fc6(out2)


        out = torch.cat((out1, out2), dim=-1)


        out = self.fc7(out)

        out3 = self.act(g)
        out3 = out3.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        out3 = self.fc2(out3)
        out3 = out3.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        out3 = self.drop_path(out3)
        out = out3 + out + shortcut
        # out = out3 + out

        out = out.permute(0, 3, 1, 2)


        return out


'''------------------------------------FANConv------------------------------------------------'''


class FANConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, bias=True, with_gate=True, pool=False):
        super(FANConv, self).__init__()
        self.input_linear_p = nn.Conv2d(input_dim, output_dim // 4, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        self.input_linear_g = nn.Conv2d(input_dim, (output_dim - output_dim // 2), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.activation = nn.GELU()
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
        if pool:
            # self.pool = nn.AdaptiveAvgPool2d(1)
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, src):
        if not hasattr(self, 'pool'):
            src = src
        else:
            src = self.pool(src)
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=1)
        return output


'''----------------------------------------LPM_Net--------------------------------------------'''


class LPM_Net(nn.Module):
    def __init__(self, num_classes, input_channels=3, patch_size=16, img_size=352, deep_supervision=False, in_chans=3,
                 embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        # self.filters = [8, 16, 32, 64, 128]        #T
        self.filters = [16, 32, 128, 160, 256]  # S
        # self.filters = [32, 64, 128, 256, 512]     #B
        # self.filters = [64, 128, 256, 512, 1024]  # L

        self.sizes = [img_size // 2, img_size // 4, img_size // 8, img_size // 16, img_size // 32]

        self.Convstage1 = Convblock(input_channels, self.filters[0])
        self.Convstage2 = Convblock(self.filters[0], self.filters[1])
        self.Convstage3 = Convblock(self.filters[1], self.filters[2])
        self.Convstage4 = Convblock(self.filters[2], self.filters[3])
        self.Convstage5 = Convblock(self.filters[3], self.filters[4])

        self.Upstage1 = FAN_Upblock(self.filters[4], self.filters[3])
        self.Upstage2 = FAN_Upblock(self.filters[3], self.filters[2])
        self.Upstage3 = FAN_Upblock(self.filters[2], self.filters[1])
        self.Upstage4 = FAN_Upblock(self.filters[1], self.filters[0])
        self.Upstage5 = FAN_Upblock(self.filters[0], self.filters[0])

        self.final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

        self.Gated1 = PIMM(self.filters[0])
        self.Gated2 = PIMM(self.filters[1])
        self.Gated3 = PIMM(self.filters[2])
        self.Gated4 = PIMM(self.filters[3])
        self.Gated5 = PIMM(self.filters[4])

        self.Gated6 = PIMM(self.filters[3])
        self.Gated7 = PIMM(self.filters[2])
        self.Gated8 = PIMM(self.filters[1])
        self.Gated9 = PIMM(self.filters[0])
        # self.Gated10 = GatedCNNBlock4(self.filters[0])

        self.final1 = nn.Conv2d(self.filters[3], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(self.filters[2], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(self.filters[1], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        #########################################
        ## Stage 1
        out = self.Convstage1(x)
        out = self.Gated1(out)
        t1 = out

        ### Stage 2
        out = self.Convstage2(out)
        out = self.Gated2(out)
        t2 = out

        ### Stage 3
        out = self.Convstage3(out)
        out = self.Gated3(out)
        # out = self.Gated3(out)
        t3 = out

        ### Stage 4
        out = self.Convstage4(out)
        out = self.Gated4(out)
        # out = self.Gated4(out)
        t4 = out

        ### Bottleneck(5)
        out = self.Convstage5(out)
        out = self.Gated5(out)
        # out = self.Gated5(out)

        ###decoder stage
        ### Stage 4
        out = self.Upstage1(out)
        out = self.Gated6(out)
        out1 = self.final1(out)
        # out = self.Gated6(out)
        out = torch.add(out, t4)


        ### Stage 3
        out = self.Upstage2(out)
        out = self.Gated7(out)
        out2 = self.final2(out)
        # out = self.Gated7(out)
        out = torch.add(out, t3)


        ### Stage 2
        out = self.Upstage3(out)
        out = self.Gated8(out)
        out3 = self.final3(out)
        out = torch.add(out, t2)


        ### Stage 1
        out = self.Upstage4(out)
        out = self.Gated9(out)
        out4 = self.final4(out)
        out = torch.add(out, t1)


        ### Stage 0
        out = self.Upstage5(out)
        out = self.final(out)
        # print(out.shape)
        return out, out1, out2, out3, out4



from ptflops import get_model_complexity_info
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LPM_Net(1).to(device)
flops, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=False)

print('flops: ', flops, 'params: ', params)


