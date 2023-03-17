# CSAB code gently borrowed and modified from: https://github.com/luuuyi/CBAM.PyTorch
import torch
import torch.nn as nn
import math

import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
        
class CSAB_v0(nn.Module):
    def __init__(self, in_planes):
        super(CSAB_v0, self).__init__()
        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention()

    def forward(self, input):
        input = self.channel_attention(input) * input
        input = self.spatial_attention(input) * input

        return input
        

class CSAB(nn.Module):
    def __init__(self, in_planes):
        """
        Modified Channel-Spatial Attention block that generates masks based on two feature inputs.
        :param in_planes: number of channels for the inputs
        """
        super(CSAB, self).__init__()
        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention()
        self.cat_conv = nn.Sequential(nn.Conv2d(2*in_planes, in_planes, kernel_size=3, padding=1),
                                      nn.ReLU())

    def forward(self, input, guide):
        input = F.interpolate(input, scale_factor=2, align_corners=False, mode='bilinear')
        att_feat = self.cat_conv(torch.cat([input, guide], dim=1))
        guide_refined = self.channel_attention(att_feat) * guide
        guide_refined = self.spatial_attention(att_feat) * guide_refined

        return guide_refined


def get_image_gradients(image: torch.Tensor, step:int=1):
    """Returns image gradients (dy, dx) for each color channel, using
    the finite-difference approximation.
    Places the gradient [ie. I(x+1,y) - I(x,y)] on the base pixel (x, y).
    Both output tensors have the same shape as the input: [b, c, h, w].
    Arguments:
        image: Tensor with shape [b, c, h, w].
        step: the size of the step for the finite difference
    Returns:
        Pair of tensors (dy, dx) holding the vertical and horizontal
        image gradients (ie. 1-step finite difference). To match the
        original size image, for example with step=1, dy will always
        have zeros in the last row, and dx will always have zeros in
        the last column.
    """
    right = F.pad(image, (0, step, 0, 0))[..., :, step:]
    bottom = F.pad(image, (0, 0, 0, step))[..., step:, :]

    dx, dy = right - image, bottom - image

    dx[:, :, :, -step:] = 0
    dy[:, :, -step:, :] = 0

    return dx, dy


class GradientLoss(nn.Module):
    """
    Gently borrowed from:
    https://github.com/victorca25/traiNNer/blob/master/codes/models/modules/loss.py
    """
    def __init__(self, ):
        super(GradientLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x:torch.Tensor, y:torch.Tensor)-> torch.Tensor:
        inputdy, inputdx = get_image_gradients(x)
        targetdy, targetdx = get_image_gradients(y)
        return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy))/2
