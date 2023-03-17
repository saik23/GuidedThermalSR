"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
from collections import OrderedDict
from typing import Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm

from pac import PacConvTranspose2d
from utils import *

def convert_to_single_channel(x):
    bs, ch, h, w = x.shape
    if ch != 1:
        x = x.reshape(bs * ch, 1, h, w)
    return x, ch


def recover_from_single_channel(x, ch):
    if ch != 1:
        bs_ch, _ch, h, w = x.shape
        assert _ch == 1
        assert bs_ch % ch == 0
        x = x.reshape(bs_ch // ch, ch, h, w)
    return x


def repeat_for_channel(x, ch):
    if ch != 1:
        bs, _ch, h, w = x.shape
        x = x.repeat(1, ch, 1, 1).reshape(bs * ch, _ch, h, w)
    return x


def th_rmse(pred, gt):
    return (pred - gt).pow(2).mean(dim=3).mean(dim=2).sum(dim=1).sqrt().mean()


def th_epe(pred, gt, small_flow=-1.0, unknown_flow_thresh=1e7):
    pred_u, pred_v = pred[:, 0].contiguous().view(-1), pred[:, 1].contiguous().view(-1)
    gt_u, gt_v = gt[:, 0].contiguous().view(-1), gt[:, 1].contiguous().view(-1)
    if gt_u.abs().max() > unknown_flow_thresh or gt_v.abs().max() > unknown_flow_thresh:
        idx_unknown = ((gt_u.abs() > unknown_flow_thresh) + (gt_v.abs() > unknown_flow_thresh)).nonzero()[:, 0]
        pred_u[idx_unknown] = 0
        pred_v[idx_unknown] = 0
        gt_u[idx_unknown] = 0
        gt_v[idx_unknown] = 0
    epe = ((pred_u - gt_u).pow(2) + (pred_v - gt_v).pow(2)).sqrt()
    if small_flow >= 0.0 and (gt_u.abs().min() <= small_flow or gt_v.abs().min() <= small_flow):
        idx_valid = ((gt_u.abs() > small_flow) + (gt_v.abs() > small_flow)).nonzero()[:, 0]
        epe = epe[idx_valid]
    return epe.mean()


def th_psnr(pred, gt):
    return tm.functional.peak_signal_noise_ratio(pred, gt)


def th_ssim(pred, gt):
    return tm.functional.structural_similarity_index_measure(pred, gt)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv0 = conv1x1(in_channels, out_channels, stride)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        residual = self.conv0(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class PacJointUpAtt(nn.Module):
    """
    PAC Joint up-sample with Attention based feature alignment.
    """
    def __init__(self, args, factor, channels=1, guide_channels=3,n_f_layers=2,
                 k_ch=32, f_sz_1=5, f_sz_2=5, u_bn=False, f_bn=False):
        super(PacJointUpAtt, self).__init__()
        self.channels = channels
        self.guide_channels = guide_channels
        self.factor = factor
        n_t_layers = n_g_layers = args.num_layers
        n_t_filters = n_g_filters = n_f_filters = args.num_channels
        self.branch_t = None
        self.branch_g = None
        self.branch_f = None
        self.k_ch = k_ch
        self.interpolation_mode = args.interpolation

        assert n_g_layers >= 1, 'Guidance branch should have at least one layer'
        assert n_f_layers >= 1, 'Final prediction branch should have at least one layer'
        assert math.log2(factor) % 1 == 0, 'factor needs to be a power of 2'
        assert f_sz_1 % 2 == 1, 'filter size needs to be an odd number'
        num_ups = int(math.log2(factor))  # number of 2x up-sampling operations
        pad = int(f_sz_1 // 2)

        if type(n_t_filters) == int:
            n_t_filters = (n_t_filters,) * n_t_layers
        else:
            assert len(n_t_filters) == n_t_layers

        if type(n_g_filters) == int:
            n_g_filters = (n_g_filters,) * (n_g_layers - 1)
        else:
            assert len(n_g_filters) == n_g_layers - 1

        if type(n_f_filters) == int:
            n_f_filters = (n_f_filters,) * (n_f_layers + num_ups - 1)
        else:
            assert len(n_f_filters) == n_f_layers + num_ups - 1

        # target branch
        t_layers = []
        n_t_channels = (channels,) + n_t_filters
        for l in range(n_t_layers):
            t_layers.append(('res{}'.format(l + 1), ResidualBlock(n_t_channels[l], n_t_channels[l + 1])))
        self.branch_t = nn.Sequential(OrderedDict(t_layers))

        # guidance branch
        g_layers = []
        n_g_channels = (guide_channels,) + n_g_filters + (n_g_filters[0],)
        for l in range(n_g_layers):
            g_layers.append(('res{}'.format(l + 1), ResidualBlock(n_g_channels[l], n_g_channels[l + 1])))
        self.branch_g = nn.Sequential(OrderedDict(g_layers))
        self.branch_g1 = nn.Sequential(conv3x3(n_g_filters[0], n_g_filters[0], 2),
                                       nn.ReLU())
        self.branch_g2 = nn.Sequential(conv3x3(n_g_filters[0], n_g_filters[0], 2),
                                       nn.ReLU())

        # attention layers
        self.att = nn.ModuleList()
        for i in range(num_ups):
            self.att.append(CSAB(n_g_filters[0]))

        # upsampling layers
        p, op = int((f_sz_2 - 1) // 2), (f_sz_2 % 2)
        self.up_convts = nn.ModuleList()
        self.up_bns = nn.ModuleList()
        n_f_channels = (n_t_channels[-1],) + n_f_filters + (channels,)
        for l in range(num_ups):
            self.up_convts.append(PacConvTranspose2d(n_f_channels[l], n_f_channels[l + 1],
                                                     kernel_size=f_sz_2, stride=2, padding=p, output_padding=op))
            if u_bn:
                self.up_bns.append(nn.BatchNorm2d(n_f_channels[l + 1]))

        # final prediction branch
        f_layers = []
        for l in range(n_f_layers):
            # TODO: Remove BN and bias in final conv layer.
            f_layers.append(('conv{}'.format(l + 1), nn.Conv2d(n_f_channels[l + num_ups], n_f_channels[l + num_ups + 1],
                                                               kernel_size=f_sz_1, padding=pad, bias=False)))
            # kernel_size=f_sz_1, padding=pad, bias=True)))
            if f_bn and l < n_f_layers - 1:
                # if f_bn:
                f_layers.append(('bn{}'.format(l + 1), nn.BatchNorm2d(n_f_channels[l + num_ups + 1])))
            if l < n_f_layers - 1:
                f_layers.append(('relu{}'.format(l + 1), nn.ReLU()))
        self.branch_f = nn.Sequential(OrderedDict(f_layers))
        print("Using Residual Block PAC Attention Upsample model")

    def forward(self, target_low, guide):
        # Ablation study to check baseline bilinear interpolation performance.
        # return F.interpolate(target_low, scale_factor=self.factor, mode='bilinear', align_corners=False)

        target_low, ch0 = convert_to_single_channel(target_low)
        x = self.branch_t(target_low)
        guide = self.branch_g(guide)
        guide1 = self.branch_g1(guide)
        guide2 = self.branch_g2(guide1)
        guides = [guide2, guide1, guide]

        for i in range(len(self.up_convts)):
            guide_cur = guides[i]
            # Attention based guide-input feature alignment correction
            guide_cur = self.att[i](x, guide_cur)
            x = self.up_convts[i](x, guide_cur)
            if self.up_bns:
                x = self.up_bns[i](x)
            x = F.relu(x)
        x = self.branch_f(x)

        if self.interpolation_mode == 'bilinear':
            x += F.interpolate(target_low, scale_factor=self.factor, align_corners=False, mode='bilinear')
        else:
            x += F.interpolate(target_low, scale_factor=self.factor, align_corners=False, mode='bicubic')
        x = recover_from_single_channel(x, ch0)

        return x
