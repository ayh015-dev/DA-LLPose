# ------------------------------------------------------------------------------
# The code is based on 2PCNet.
# (https://github.com/mecarill/2pcnet)
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from numpy import random as R
import cv2

class ELLA():
    def __init__(self, cfg):
        self.adjust = cfg.DATASET.ADJUST_ELLA
        self.p = 0.0 if self.adjust else 0.5

    def mask_img(self, img, cln_img):
        while R.random() > 0.4:       
            x1 = R.randint(img.shape[1])
            x2 = R.randint(img.shape[1])
            y1 = R.randint(img.shape[2])
            y2 = R.randint(img.shape[2])
            img[:, x1:x2, y1:y2] = cln_img[:, x1:x2, y1:y2]

        return img

    def aug(self, x):

        img = x

        cln_img_zero = img.detach().clone()

         # Gamma
        if R.random() > self.p:
            cln_img = img.detach().clone()
            val = 1/(R.random() * 0.3 + 0.2)
            img = T.functional.adjust_gamma(img, val)
            img = self.mask_img(img, cln_img)
        
        # Brightness
        if R.random() > self.p:
            cln_img = img.detach().clone()
            val = R.random() * 0.04 + 0.01
            img = T.functional.adjust_brightness(img, val)
            img = self.mask_img(img, cln_img)
        
        # Contrast
        if R.random() > 0.5:
            cln_img = img.detach().clone()
            val = R.random() * 0.8 + 0.2
            img = T.functional.adjust_contrast(img, val)
            img = self.mask_img(img, cln_img)
        img = self.mask_img(img, cln_img_zero)
        
        # Noise
        if R.random() > 0.5:
            n = torch.clamp(torch.normal(0, R.random() * 40, img.shape), min=0)
            img = n + img
            img = torch.clamp(img, max=255).type(torch.uint8)

        return img






