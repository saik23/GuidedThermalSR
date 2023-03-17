"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import os.path
import random
import math
import collections
import random

import numpy as np
import torch as th
from torch.utils.data import Dataset
import torchvision.transforms as tt
from PIL import Image

"""
PBVS Thermal Super Resolution - Track 2 dataset.
Dataset contains a high resolution RGB image and a corresponding low resolution thermal image (x8).
Ground truth is a high resolution thermal image.
"""
class PBVS2(Dataset):
    def __init__(self, root, split, file_names_path, transform=None):
        super(PBVS2, self).__init__()
        self.dir = root
        self.mode = split
        self.transform = transform

        if self.mode == 'train':
            self.input_dir = os.path.join(self.dir, 'train/input')
            self.label_dir = os.path.join(self.dir, 'train/label')
            self.guide_dir = os.path.join(self.dir, 'train/guide')
        elif self.mode == 'val':
            self.input_dir = os.path.join(self.dir, 'val/input')
            self.label_dir = os.path.join(self.dir, 'val/label')
            self.guide_dir = os.path.join(self.dir, 'val/guide')
        else:
            self.input_dir = os.path.join(self.dir, 'test/input')
            self.label_dir = os.path.join(self.dir, 'test/label')     # Dummy label created by lancosz interpolation.
            self.guide_dir = os.path.join(self.dir, 'test/guide')

        file_names_path=os.path.join(root, file_names_path)
        with open(file_names_path) as fn:
            self.filenames = fn.read().splitlines()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # filepaths
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        guide_path = os.path.join(self.guide_dir, self.filenames[idx])
        label_path = os.path.join(self.label_dir, self.filenames[idx])

        # input
        input = np.array(Image.open(input_path), dtype=np.uint8)[:,:,0]     # 80x60 image
        label = np.array(Image.open(label_path), dtype=np.uint8)     # 640x480 image
        guide = np.array(Image.open(guide_path).convert('RGB'), dtype=np.uint8)     # 640x480x3 image
        guide = guide.transpose(2, 0, 1)     # 3x640x480 image

        if self.transform is not None:
            input, guide, label = self.transform((input, guide, label))

        return input, guide, label, self.filenames[idx]


class AssembleJointUpsamplingInputs(object):
    def __init__(self, crop=None, flip=True, rotate=True):
        self.crop = crop if (isinstance(crop, collections.Iterable) or crop is None) else (crop, crop)
        self.flip = flip
        self.rotate = rotate

    @staticmethod
    def random_crop_params(img, output_size):
        """Get parameters (i, j, h, w) for random crop.
        Adapted from torchvision.transforms.transforms.RandomCrop.get_params
        """
        h, w = img.shape[1:]
        out_h, out_w = output_size
        if w == out_w and h == out_h:
            return 0, 0, h, w

        i = random.randint(0, h - out_h)
        j = random.randint(0, w - out_w)
        return i, j, out_h, out_w

    def __call__(self, input_guide_target_tuple):
        input, guide, target = input_guide_target_tuple
        if input.ndim == 2:
            input = input.reshape((1, ) + input.shape)
        if target.ndim == 2:
            target = target.reshape((1, ) + target.shape)

        if self.crop is not None:
            # Get the random crop parameters from the input size.
            # Use the random crop parameter from the input to crop the guide and the target
            i, j, out_h, out_w = self.random_crop_params(input, self.crop)
            input = input[:, i:i+out_h, j:j+out_w]
            i, j, out_h, out_w = 8*i, 8*j, 8*out_h, 8*out_w
            guide = guide[:, i:i+out_h, j:j+out_w]
            target = target[:, i:i+out_h, j:j+out_w]

        # Horizontal flip
        if self.flip and random.random() < 0.5:
            input = input[:, :, ::-1]
            guide = guide[:, :, ::-1]
            target = target[:, :, ::-1]

        # Vertical flip
        if self.flip and random.random() < 0.5:
            input = input[:, ::-1, :]
            guide = guide[:, ::-1, :]
            target = target[:, ::-1, :]

        # TODO: Add jitter, and other augmentations.
        # TODO: Check if making copies is to have structured data?
        input = input.copy()
        guide = guide.copy()
        target = target.copy()

        input = th.from_numpy(input).float().div(255)
        guide = th.from_numpy(guide).float().div(255)
        target = th.from_numpy(target).float().div(255)

        if self.rotate:
            random_angle=random.choice([0,90,180])
            input = tt.functional.rotate(input, random_angle)
            guide = tt.functional.rotate(guide, random_angle)
            target = tt.functional.rotate(target, random_angle)

        """
        print("input minmax:", th.min(input), th.max(input))
        print("guide minmax:", th.min(guide), th.max(guide))
        print("label minmax:", th.min(target), th.max(target))
        """
        return input, guide, target
