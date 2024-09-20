import numpy as np
import torch
import cv2
import copy

def deprocess(img):
    img = np.transpose(img, [1,2,0])
    img = img * np.float32([0.229, 0.224, 0.225])
    img = img + np.float32([0.485, 0.456, 0.406])
    img = img * 255
    img = np.uint8(img)
    return img

def vis_kpt(img, hmap, minmax=False):
    img = img.copy()
    if minmax:
        hmap = hmap - hmap.min()
        hmap = hmap / hmap.max()
    else:
        hmap = np.clip(hmap, 0.0, 1.0)
    hmap = np.uint8(hmap * 255)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    res = cv2.addWeighted(hmap, 0.7, img, 0.3, 0)
    return res

def vis_one(img, hmap, minmax=False):
    img = deprocess(img)
    img = cv2.resize(img, (hmap.shape[-1], hmap.shape[-2]))
    if len(hmap.shape)==3:
        res = [vis_kpt(img, hmap[i], minmax=minmax) for i in range(hmap.shape[0])]
    else:
        res = [vis_kpt(img, hmap)]
    return res

def vis_batch(imgs, hmaps, outdir=None, minmax=False):
    hmaps = hmaps.cpu().detach().numpy()
    imgs = imgs.cpu().detach().numpy()
    res = []
    for i in range(imgs.shape[0]):
        res.append(vis_one(imgs[i], hmaps[i], minmax=minmax))
    h = len(res)
    w = len(res[0])
    s_w = hmaps.shape[-1]
    s_h = hmaps.shape[-2]
    canvas = np.zeros([h * s_h, w * s_w, 3], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            canvas[i*s_h:i*s_h+s_h, j*s_w:j*s_w+s_w] = res[i][j]
    if outdir:
        cv2.imwrite(outdir, canvas)
    return canvas

def offset_one(img, offset, pt):
    s = offset.shape[-1]
    img = deprocess(img)
    img = cv2.resize(img, (s, s))
    res = []
    for j in range(config.num_pts):
        canvas = np.zeros([s, s], dtype=np.uint8)
        for i in range(pt.shape[0]):
            if pt[i,j,2]<=0:
                continue
            xx = int(pt[i,-1,0])
            yy = int(pt[i,-1,1])
            # print(xx, yy)
            if xx<0 or xx>=s or yy<0 or yy>=s:
                continue
            dx = offset[j*2, yy, xx]
            dy = offset[j*2+1, yy, xx]
            # print(dx, dy)
            # dx = dy = 0
            x = int(xx + dx)
            y = int(yy + dy)
            # print(x,y)
            for xx in range(x-2,x+3):
                for yy in range(y-2,y+3):
                    if xx>=0 and xx<s and yy>=0 and yy<s:
                        canvas[yy,xx] = 180
        canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
        canvas = cv2.addWeighted(canvas, 0.7, img, 0.3, 0)
        res.append(canvas)
    res = np.concatenate(res, axis=1)
    return res

def vis_offset(imgs, offsets, pts, outdir=None):
    imgs = imgs.cpu().detach().numpy()
    offsets = offsets.cpu().detach().numpy()
    pts = pts.cpu().detach().numpy()
    res = []
    for i in range(imgs.shape[0]):
        res.append(offset_one(imgs[i], offsets[i], pts[i]))
    res = np.concatenate(res, axis=0)
    if outdir:
        cv2.imwrite(outdir, res)
    return res

EDGES = [([8, 10], [255, 0, 0]),  # l_ankle -> l_knee
                       ([6, 8], [155, 85, 0]),  # l_knee -> l_hip
                    #    ([11, 5],  [155, 85, 0]),  # l_hip -> l_shoulder
                       ([7, 9], [0, 0, 255]),  # r_hip -> r_knee
                       ([9, 11], [17, 25, 10]),  # r_knee -> r_ankle
                    #    ([12, 6],  [0, 0, 255]),  # r_hip  -> r_shoulder
                    #    ([3, 1],   [0, 255, 0]),  # l_ear -> l_eye
                    #    ([1, 2],   [0, 255, 5]),  # l_eye -> r_eye
                    #    ([1, 0],   [0, 255, 170]),  # l_eye -> nose
                    #    ([0, 2],   [0, 255, 25]),  # nose -> r_eye
                    #    ([2, 4],   [0, 17, 255]),  # r_eye -> r_ear
                       ([2, 4],   [0, 220, 0]),  # l_wrist -> l_elbow
                       ([0, 2],   [0, 220, 0]),  # l_elbow -> l_shoulder
                    #    ([5, 6],   [125, 125, 155]), # l_shoulder -> r_shoulder
                       ([1, 3],   [25, 0, 55]),  # r_shoulder -> r_elbow
                       ([3, 5], [25, 0, 255]),  # r_elbow -> r_wrist
                       ([12, 13], [155, 85, 0]), # head -> neck
                       ([13, 0], [0, 0, 255]),   # neck -> l_shoulder 
                       ([13, 1], [0, 255, 0]),  # neck -> r_shoulder
                       ([13, 7], [0, 255, 170]), # neck -> r_hip
                       ([13, 6], [0, 255, 25]), # neck -> l_hip
                       ]  

def vis_single_pose(img, pose, outdir=None):
    canvas = copy.deepcopy(img)
    #print(pose[15])
    for j in range(len(EDGES)):
        current_line = EDGES[j]
        start_point_idx = current_line[0][0]
        end_point_idx = current_line[0][1]
        color = current_line[1]
        #print(start_point_idx)
        start_point_x, start_point_y, start_point_vis = pose[start_point_idx]
        end_point_x, end_point_y, end_point_vis = pose[end_point_idx]

        start_point_x, start_point_y = int(start_point_x), int(start_point_y)
        end_point_x, end_point_y = int(end_point_x), int(end_point_y)

        if start_point_x == 0 and start_point_y == 0:
            if end_point_x != 0 or end_point_y != 0:
                    cv2.circle(canvas, (end_point_x, end_point_y), 2, [255, 0, 0], 2)
            continue

        if end_point_x == 0 and end_point_y == 0:
            if start_point_x != 0 or start_point_y != 0:
                    cv2.circle(canvas, (start_point_x, start_point_y), 2, [255, 0, 0], 2)
            continue

        if start_point_vis != 0 and end_point_vis != 0:
            cv2.line(canvas, (start_point_x, start_point_y), (end_point_x, end_point_y), color, thickness=2,
                         lineType=8)

    cv2.imwrite(outdir, canvas)

def vis_detected_joints(img, joints_with_types, outdir):
    canvas = copy.deepcopy(img)
    for joints in joints_with_types:
        for joint in joints:
            if joint[2] > 0.:
                cv2.circle(canvas, (int(joint[0]) * 4, int(joint[1]) * 4) , 2, [255, 0, 0], 2) # pay attention to *4

    cv2.imwrite(outdir, canvas)