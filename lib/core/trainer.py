from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import copy
import time

import torch
import torch.nn as nn
import numpy as np
import numpy.random as R

from models.hrnet_main import get_pose_net as get_pose_net_main
from models.hrnet_comp import get_pose_net as get_pose_net_comp

from core.nms import nms_core
from core.group import HeatmapParser
from core.loss_main import MultiLossFactory as loss_main
from core.inference import get_high_confidence_heatmap
from dataset.target_generators import WeightsmapGenerator, OffsetGenerator

from utils.utils import AverageMeter

def do_PT_train(cfg, rank,model, data_loader, loss_factory, optimizer, epoch):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    if 'main' in cfg.MODEL.NAME:
       offset_loss_meter = AverageMeter()
    elif 'comp' in cfg.MODEL.NAME:
       push_loss_meter = AverageMeter()
       pull_loss_meter = AverageMeter()

    model.train()

    end = time.time()
    for i, (real_WL, fake_LL, WL_GTs, _, _, WL_joints) in enumerate(data_loader):
        data_time.update(time.time() - end)
        if 'PT_WL' in cfg.TRAIN.STAGE:
            image = real_WL.cuda(non_blocking=True)
            pheatmap, poffset_tag= model(image)
        elif 'PT_LL' in cfg.TRAIN.STAGE:
            image = fake_LL.cuda(non_blocking=True)
            pheatmap, poffset_tag= model(image)
    
        heatmap, mask, offset, offset_w = WL_GTs
        heatmap = heatmap.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True)
        offset_w = offset_w.cuda(non_blocking=True)
        WL_joints = WL_joints.cuda(non_blocking=True)
     
        if 'main' in cfg.MODEL.NAME:
            heatmap_loss, offset_loss = loss_factory(pheatmap, poffset_tag, heatmap, mask, offset, offset_w)
            loss = 0
            if heatmap_loss is not None:
               heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
               loss = loss + heatmap_loss
            if offset_loss is not None:
               offset_loss_meter.update(offset_loss.item(), image.size(0))
               loss = loss + offset_loss
        elif 'comp' in cfg.MODEL.NAME:
            heatmap_loss, push_loss, pull_loss = loss_factory(pheatmap, heatmap[:, :-1], mask[:, :-1], poffset_tag, WL_joints.long())
            loss = 0
            if heatmap_loss is not None:
               heatmap_loss_meter.update(heatmap_loss.item(), image.size(0))
               loss = loss + heatmap_loss
            if push_loss is not None:
               push_loss_meter.update(push_loss.item(), image.size(0))
               loss = loss + push_loss
            if pull_loss is not None:
               pull_loss_meter.update(pull_loss.item(), image.size(0))
               loss = loss + pull_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % cfg.PRINT_FREQ == 0 and rank == 0: 
            if 'main' in cfg.MODEL.NAME:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmap_loss_meter, 'heatmaps'),
                      offset_loss=_get_loss_info(offset_loss_meter, 'offsets'),
                  )
            elif 'comp' in cfg.MODEL.NAME:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{push_loss}{pull_loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time,
                      heatmaps_loss=_get_loss_info(heatmap_loss_meter, 'heatmaps'),
                      push_loss=_get_loss_info(push_loss_meter, 'push'),
                      pull_loss=_get_loss_info(pull_loss_meter, 'pull'),
                  )
            logger.info(msg)

class KnowledgeAcquisition(nn.Module):
    def __init__(self, cfg):
        super(KnowledgeAcquisition, self).__init__()
        self.cfg = cfg
        
        self.main = get_pose_net_main(cfg)
        for param in self.main.parameters():
            param.requires_grad = False
        self.comp = get_pose_net_comp(cfg)
        for param in self.comp.parameters():
            param.requires_grad = False
        self.student = get_pose_net_main(cfg, is_train=True)

        self.loss_factory = loss_main(cfg)

        self.weightsmap_generator = WeightsmapGenerator(cfg.DATASET.OUTPUT_SIZE, 15)
        self.offset_generator = OffsetGenerator(
            cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
            cfg.DATASET.NUM_JOINTS, cfg.DATASET.OFFSET_RADIUS) 
        self.parser = HeatmapParser(cfg)
    
    def offset_to_scale(self, off_map):
        '''
        Args:
            off: b x (k*2) x h x w

        Returns:
            scale_map : b x 1 x h x w
        '''
        off_map_list = off_map.split(2, dim=1) # [b x 2 x h x w, b x 2 x h x w ...] k
        off_map_new = torch.stack(off_map_list, dim=1)  # b x k x 2 x h x w
        off_map_x = off_map_new[:, :, [0], :, :]  # b x k x 1 x h x w
        off_map_y = off_map_new[:, :, [1], :, :]  # b x k x 1 x h x w

        max_x, _ = torch.max(off_map_x, dim=1)
        min_x, _ = torch.min(off_map_x, dim=1)
        x = max_x - min_x

        max_y, _ = torch.max(off_map_y, dim=1)
        min_y, _ = torch.min(off_map_y, dim=1)
        y = max_y - min_y

        scale_map = 1. / (torch.sqrt(x * x + y * y) * 1.1) # enlarge 10% area
        return scale_map
    
    def generate_main_teacher_poses(self, centers, offsets):
        '''
        centers: n x 1 x 3
        offsets: k*2 x h x w
        '''
        
        center_x = centers[:, 0, 0] # n
        center_y = centers[:, 0, 1] # n

        offset = offsets[:, center_y.long(), center_x.long()].permute(1, 0) # n x k*2
        offset_list = offset.split(2, dim=-1)
        offset = torch.stack(offset_list, dim=1) # n x k x 2
        locations = centers[:,:,:2] - offset #n x k x 2
        
        n, k, _ = locations.shape
        fake_scores = torch.ones(n, k, 1).to(offset.device) # n, k, 1
        locations = torch.cat([locations, fake_scores], dim=-1) # n, k, 3
        
        poses_with_centers = torch.cat([locations, centers], dim=1) # n, k+1, 3
        return poses_with_centers
    
    def poses2bbox(self, poses, h, w, expand_factor=1.5):
            
        x = poses[:, :, 0] * 4  # n, k
        y = poses[:, :, 1] * 4  # n, k
        v = poses[:, :, 2]
            
        x_nonzero = copy.deepcopy(x)
        y_nonzero = copy.deepcopy(y)

        x_nonzero[v == 0.] = torch.inf
        y_nonzero[v == 0.] = torch.inf 

        x_min, x_max = torch.min(x_nonzero, dim=-1)[0], torch.max(x, dim=-1)[0] # n
        y_min, y_max = torch.min(y_nonzero, dim=-1)[0], torch.max(y, dim=-1)[0] # n
        bbox_h, bbox_w = x_max - x_min, y_max - y_min
            
        factor = (expand_factor - 1.0) / 2.
        x_min, x_max = x_min - bbox_w * factor, x_max + bbox_w * factor
        y_min, y_max = y_min - bbox_h * factor, y_max + bbox_h * factor

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        area = bbox_w * bbox_w + bbox_h * bbox_h
        return torch.clip(x_min, 0., w-1), torch.clip(x_max, 0., w-1), torch.clip(y_min, 0., h-1), torch.clip(y_max, 0., h-1), area
    
    def PDA(self, img, poses_with_centers):
        '''
        img: 3, h, w
        poses_with_centers: n, k, (x,y,conf)
        '''
        _, h, w = img.shape
        x_min, x_max, y_min, y_max, area = self.poses2bbox(poses_with_centers, h, w)
        for _, (x1, x2, y1, y2) in enumerate(zip(x_min, x_max, y_min, y_max)):
            
            img_crop = copy.deepcopy(img[:, y1.long():y2.long(), x1.long():x2.long()])
            if R.random() > 0.5:
               img_crop = self.adjust_darkness(img_crop)
            img[:, y1.long():y2.long(), x1.long():x2.long()] = img_crop
        
        return img, area
    
    def adjust_darkness(self, img):
    
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).to(img.device)
        denormal_img = img * std + mean
        denormal_img *= 255.
    
        percent = R.random() * 0.4 + 0.1

        denormal_img *= percent
        denormal_img /= 255.
        img = (denormal_img - mean) / std

        return img
    
    def pesudo_label_generation(self, oob_mask, main_heat, offsets, comp_heat, tags, real_LL):
        device = main_heat.device

        # pesudo label generation
        _, c_heat, h, w = main_heat.shape
        _, c_off, h, w = offsets.shape
        real_LL_degraded_list, heatmap_list, heatmap_weights_list, offmap_list, off_weights_list = [], [], [], [], []
        for i, (img, cen, off, heat, tag) in enumerate(zip(real_LL, main_heat[:,[-1]], offsets, comp_heat, tags)):
            with torch.no_grad():
                # main teacher
                main_centers, main_scores = get_high_confidence_heatmap(self.cfg, cen, thres=0.9) 
                main_centers = torch.cat(((main_centers % w).unsqueeze(1),
                                        (main_centers // w).unsqueeze(1),
                                        main_scores.unsqueeze(1)), dim=1)      
                centers = main_centers.unsqueeze(1)
                if centers.shape[0] != 0:
                   poses_main = self.generate_main_teacher_poses(centers, off)
                else:
                   poses_main = torch.zeros((0, 15, 3)).to(device)

                # complementary teacher
                heat = heat.unsqueeze(0)
                tag = tag.unsqueeze(0)
                poses_comp_list, scores_list = self.parser.parse(heat, tag, adjust=True, refine=False)
                
                if len(scores_list) != 0:
                   poses_comp = torch.from_numpy(poses_comp_list[0]).to(device)
                   scores_comp = torch.from_numpy(np.stack(scores_list, axis=0)).to(device)

                   selected_idx = torch.where(scores_comp > 0.5)[0]
                   poses_comp = poses_comp[selected_idx]
                   scores_comp = scores_comp[selected_idx]
                   
                   # keep the poses with at least half body
                   keypoints_num = torch.count_nonzero(poses_comp[:, :, -2], dim=-1)
                   keep_idx = torch.where(keypoints_num >= 10)[0]
                   poses_comp = poses_comp[keep_idx]
                   scores_comp = scores_comp[keep_idx]
                else:
                   poses_comp = torch.zeros((0, 14, 4)).to(device)
                   scores_comp = torch.zeros(0).to(device)
                
                poses_all = torch.cat([poses_main[:, :-1, :], poses_comp[:, :, :3]], dim=0)
                scores_all = torch.cat([torch.ones(poses_main.size(0)).to(device), scores_comp], dim=0)
                
                if poses_all.shape[0] == 0:
                    real_LL_degraded_list.append(img)
                    heatmap_list.append(torch.zeros(1, c_heat, h, w).to(device))
                    heatmap_weights_list.append(torch.zeros(1, c_heat, h, w).to(device))
                    offmap_list.append(torch.zeros(1, c_off, h, w).to(device))
                    off_weights_list.append(torch.zeros(1, c_off, h, w).to(device))
                    continue                

                keep_ind = nms_core(self.cfg, poses_all[:, :, :2], scores_all)
                poses_all = poses_all[keep_ind]
                scores_all = scores_all[keep_ind]
                
                num_nonzeros = torch.count_nonzero(poses_all[:, :, -1], dim=-1)
                centers = poses_all.sum(dim=-2) / num_nonzeros.unsqueeze(1)
                poses_all_with_centers = torch.cat([poses_all, centers.unsqueeze(1)], dim=1)  

                # degradation
                real_LL_degraded, area = self.PDA(img, poses_all_with_centers)
                real_LL_degraded_list.append(real_LL_degraded)
                
                # pesudo label generation
                heatmaps, heatmap_weights = self.weightsmap_generator(poses_all_with_centers.detach().cpu().numpy(), self.cfg.DATASET.SIGMA, self.cfg.DATASET.CENTER_SIGMA, 0.0)
                heatmaps = torch.from_numpy(heatmaps).to(device).unsqueeze(0)
                heatmap_weights = torch.from_numpy(heatmap_weights).to(device).unsqueeze(0) 

                offset, offset_weights = self.offset_generator(poses_all_with_centers.detach().cpu().numpy(), area.cpu().numpy())
                offset = torch.from_numpy(offset).to(device).unsqueeze(0)
                offset_weights = torch.from_numpy(offset_weights).to(device).unsqueeze(0) 

                heatmap_weights += oob_mask[i]
                heatmap_weights = torch.clip(heatmap_weights, max=1.0)
                
                heatmap_list.append(heatmaps)
                heatmap_weights_list.append(heatmap_weights)
                offmap_list.append(offset)
                off_weights_list.append(offset_weights)
        
        real_LL_degraded = torch.stack(real_LL_degraded_list, dim=0)
        heatmaps = torch.cat(heatmap_list, dim=0)
        heatmap_weights = torch.cat(heatmap_weights_list, dim=0)
        offsets = torch.cat(offmap_list, dim=0)
        offset_weights = torch.cat(off_weights_list, dim=0)
        return real_LL_degraded, heatmaps, heatmap_weights, offsets, offset_weights
            
    def forward(
            self, 
            real_LL,
            fake_LL,
            WL_GTs,
            oob_mask
            ):
        # supervised training
        WL_heatmap, WL_heatmap_w, WL_off, WL_off_w = WL_GTs
        stu_heat, stu_off = self.student(fake_LL)
        stu_heat_loss, stu_off_loss = self.loss_factory(stu_heat, stu_off, WL_heatmap, WL_heatmap_w, WL_off, WL_off_w)
        l_sup = stu_heat_loss + stu_off_loss

        # unsupervised training
        self.main.eval()
        self.comp.eval()
        with torch.no_grad():
            main_heat, main_offsets = self.main(real_LL)
            comp_heat, comp_tags = self.comp(real_LL)
            # pseudo label generations
            real_LL_degraded, heatmaps, heatmap_weights, offsets, offset_weights = self.pesudo_label_generation(oob_mask, main_heat, main_offsets, comp_heat, comp_tags, real_LL)
     
        stu_heat2, stu_off2 = self.student(real_LL_degraded)
        l_unsup_heatmap, l_unsup_offset = self.loss_factory(stu_heat2, stu_off2, heatmaps, heatmap_weights, offsets, offset_weights)
        l_unsup = l_unsup_heatmap + l_unsup_offset
           
        return l_sup, l_unsup
    
def do_KA_train(cfg, rank, model, data_loader, optimizer, loss_factory, epoch):
    logger = logging.getLogger("Training")
    
    lsup_loss_meter = AverageMeter()
    lunsup_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.module.student.train() 
    end = time.time()
    for i, (_, fake_LL, WL_GTs, real_LL, oob_mask, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        WL_GTs = [x.cuda(non_blocking=True) for x in WL_GTs]
        real_LL = real_LL.cuda(non_blocking=True)
        fake_LL = fake_LL.cuda(non_blocking=True)
        oob_mask = oob_mask.cuda(non_blocking=True)

        l_sup, l_unsup = model(real_LL, fake_LL, WL_GTs, oob_mask)

        if l_unsup == 0.:
            l_sup *= 0.0

        loss = l_sup + l_unsup
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
               
        lsup_loss_meter.update(l_sup.item(), fake_LL.size(0))
        lunsup_loss_meter.update(l_unsup.item(), real_LL.size(0))
        loss_meter.update(loss.item(), real_LL.size(0))
               
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % cfg.PRINT_FREQ == 0 and rank == 0: 
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{lsup_loss}{lunsup_loss}{loss}'.format(
                      epoch, i, len(data_loader),
                      batch_time=batch_time,
                      speed=real_LL.size(0)/batch_time.val,
                      data_time=data_time,
                      lsup_loss=_get_loss_info(lsup_loss_meter, 'lsup'),
                      lunsup_loss=_get_loss_info(lunsup_loss_meter, 'lunsup'),
                      loss=_get_loss_info(loss_meter, 'total_loss')
                  )
            logger.info(msg)

def _get_loss_info(meter, loss_name):
    msg = ''
    msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
        name=loss_name, meter=meter
    )
    return msg
