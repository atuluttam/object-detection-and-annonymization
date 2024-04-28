import argparse
import os
import platform
import shutil
import time
import pandas as pd
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from numpy import random
from utils import hyper_parameters as hp
import numpy as np
import sys
sys.path.insert(0,'./Detection')
sys.path.insert(1, './Detection/ScaledYOLOv4')
from Detection.ScaledYOLOv4.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output

class yolov4():
    def __init__(self,weights,img_size):
        self.device = torch.device('cuda:0')
        self.model = self.attempt_load(weights, map_location=self.device)  # load FP32 model
        self.img_size = check_img_size(img_size, s=self.model.stride.max())  # check img_size
        #print("img size", self.img_size )
        self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        
        print("model loaded successfully")
        
    def attempt_load(self,weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        model = Ensemble()
        for w in weights if isinstance(weights, list) else [weights]:
            model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

        if len(model) == 1:
            return model[-1]  # return model
        else:
            print('Ensemble created with %s\n' % weights)
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model  # return ensemble
        
    def letterbox(self,img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

        
    def preprocess(self,image):
        img = self.letterbox(image, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() 
#         if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    
    def detect(self,image,conf_thres=0.4,iou_thres=0.5,classes=80):
        img = self.preprocess(image)
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres,agnostic=True)
        result = []
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                     # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                   # Write results
                for *xyxy, conf, cls in det:
                    label = '%s' % (self.names[int(cls)])
                    if label in hp.acceptableCategories:
                        coord = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        x = (coord[0] + coord[2])/2
                        y = (coord[1] + coord[3])/2
                        width = coord[2] - coord[0]
                        height = coord[3] - coord[1]
                        val = (label, conf.tolist(), (x, y, width, height))
                        result.append(val)

                    # detection_list.append([class_name, round(conf.tolist(),4),(xyxy[0].tolist(),xyxy[1].tolist(),xyxy[2].tolist(),xyxy[3].tolist())])
        return result



        
    



