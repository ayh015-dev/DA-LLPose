from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import pycocotools
from .ExLPoseDataset import ExLPoseDataset

logger = logging.getLogger(__name__)

class ExLPoseKeypoints(ExLPoseDataset):
    def __init__(self, cfg, dataset, heatmap_generator=None, offset_generator=None, joints_generator=None, transforms=None, ella=None):
        super().__init__(cfg, dataset, ella)
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints + 1

        self.sigma = cfg.DATASET.SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator
        self.joints_generator = joints_generator
        self.transforms = transforms

        self.ids_WL = [
            img_id
            for img_id in self.ids_WL
            if len(self.coco_WL.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        ]
        self.ids_LL = [
            img_id
            for img_id in self.ids_LL
            if len(self.coco_LL.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        ]

    def _get_labels_per_image(self, img, img_aug, anno, img_info, need_label=True):
        mask = self.get_mask(anno, img_info)
        anno = [obj for obj in anno if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0]
        joints, area = self.get_joints(anno)

        if self.transforms:
            # generating out-of-distribution mask: 
            # masking out the region out of the image after rotation.
            oob_mask = np.uint8(np.zeros_like(img) + 255)
            img, mask_list, oob_mask, joints_list, area, img_aug = self.transforms(
                img, [mask], oob_mask, [joints], area, img_aug
            )
            oob_mask = (oob_mask[:, :, [-1]]).astype(np.float_) / 255.
            oob_mask = (~oob_mask.astype(np.bool_)).astype(np.float_)
            oob_mask = oob_mask.transpose(2, 0, 1)

            heatmap, ignored = self.heatmap_generator(
                joints_list[0], self.sigma, self.center_sigma, self.bg_weight)
            heatmap_weights = mask_list[0] * ignored  # (k+1) * h * w

            heatmap_weights += oob_mask
            heatmap_weights = np.clip(heatmap_weights, 0.0, 1.0)
            
            offset, offset_weights = self.offset_generator(joints_list[0], area)
            joints = self.joints_generator(joints_list[0][:, :-1, :]) 

        return img, img_aug, heatmap, heatmap_weights, offset, offset_weights, oob_mask, joints

    def __getitem__(self, idx):
        real_WL, fake_LL, real_LL, anno_WL, anno_LL, image_info_WL, image_info_LL=super().__getitem__(idx)

        real_WL, fake_LL, WL_heatmap, WL_heatmap_w, WL_off, WL_off_w, _, WL_joints=self._get_labels_per_image(real_WL, fake_LL, anno_WL, image_info_WL)
        WL_GTs = [WL_heatmap, WL_heatmap_w, WL_off, WL_off_w]

        real_LL, _, _, _, _, _, oob_mask, _=self._get_labels_per_image(real_LL, None, anno_LL, image_info_LL)
       
        return real_WL, fake_LL, WL_GTs, real_LL, oob_mask, WL_joints

    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints_with_center, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

            area[i, 0] = self.cal_area_2_torch(
                torch.tensor(joints[i:i+1,:,:]))
            
            joints_sum = np.sum(joints[i, :-1, :2], axis=0)
            num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
            if num_vis_joints <= 0:
                joints[i, -1, :2] = 0
            else:
                joints[i, -1, :2] = joints_sum / num_vis_joints
            joints[i, -1, 2] = 2

        return joints, area
    
    def get_mask(self, anno, img_info):
        m = np.zeros((img_info['height'], img_info['width']))
        return m < 0.5
