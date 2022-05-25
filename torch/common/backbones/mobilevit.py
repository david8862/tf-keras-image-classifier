#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates MobileViT Model as defined in:
MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
https://arxiv.org/abs/2110.02178
Modified from https://github.com/chinhsuanwu/mobilevit-pytorch
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from einops import rearrange

__all__ = ['mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs']


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, dims, channels, num_classes=1000, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        self.patch_size = patch_size

        L = [2, 4, 3]

        conv1 = conv_nxn_bn(3, channels[0], stride=2)

        mv2 = nn.ModuleList([])
        mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        mvit = nn.ModuleList([])
        mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.features = nn.Sequential(conv1,
                                      mv2[0],

                                      mv2[1],
                                      mv2[2],
                                      mv2[3],      # Repeat

                                      mv2[4],
                                      mvit[0],

                                      mv2[5],
                                      mvit[1],

                                      mv2[6],
                                      mvit[2]
                                     )

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        #self.pool = nn.AvgPool2d(ih//32, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        ih, iw = tuple(x.shape[-2:])
        ph, pw = self.patch_size
        assert ih % (ph*32) == 0 and iw % (pw*32) == 0, 'input shape should be multiples of 32*patch_size'

        x = self.features(x)
        x = self.conv2(x)
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def mobilevit_xxs(**kwargs):
    """
    Constructs a MobileViT_XXS model
    """
    if 'pretrained' in kwargs.keys():
        pretrained = kwargs['pretrained']
        kwargs.pop('pretrained')
    else:
        pretrained = None

    if 'weights_path' in kwargs.keys():
        weights_path = kwargs['weights_path']
        kwargs.pop('weights_path')
    else:
        weights_path = None

    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]

    model = MobileViT(dims, channels, expansion=2, **kwargs)

    if pretrained:
        # load pretrained weights
        weights_url = 'https://github.com/david8862/tf-keras-image-classifier/releases/download/v1.0.0/mobilevit_xxs.pth'

        pretrain_dict = model_zoo.load_url(weights_url)
        model.load_state_dict(pretrain_dict)

    if weights_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model


def mobilevit_xs(**kwargs):
    """
    Constructs a MobileViT_XS model
    """
    if 'pretrained' in kwargs.keys():
        pretrained = kwargs['pretrained']
        kwargs.pop('pretrained')
    else:
        pretrained = None

    if 'weights_path' in kwargs.keys():
        weights_path = kwargs['weights_path']
        kwargs.pop('weights_path')
    else:
        weights_path = None

    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]

    model = MobileViT(dims, channels, **kwargs)

    if pretrained:
        # load pretrained weights
        weights_url = 'https://github.com/david8862/tf-keras-image-classifier/releases/download/v1.0.0/mobilevit_xs.pth'

        pretrain_dict = model_zoo.load_url(weights_url)
        model.load_state_dict(pretrain_dict)

    if weights_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model


def mobilevit_s(**kwargs):
    """
    Constructs a MobileViT_S model
    """
    if 'pretrained' in kwargs.keys():
        pretrained = kwargs['pretrained']
        kwargs.pop('pretrained')
    else:
        pretrained = None

    if 'weights_path' in kwargs.keys():
        weights_path = kwargs['weights_path']
        kwargs.pop('weights_path')
    else:
        weights_path = None

    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]

    model = MobileViT(dims, channels, **kwargs)

    if pretrained:
        # load pretrained weights
        weights_url = 'https://github.com/david8862/tf-keras-image-classifier/releases/download/v1.0.0/mobilevit_s.pth'

        pretrain_dict = model_zoo.load_url(weights_url)
        model.load_state_dict(pretrain_dict)

    if weights_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model



if __name__ == '__main__':
    model = mobilevit_xxs(pretrained=False)

    from torchsummary import summary
    summary(model, input_size=(3, 256, 256))

    #torch.save(model, 'check.pth')
