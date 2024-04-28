#! /usr/bin/env python

import json

from keras.models import load_model

from Detection.Yolov3.utils.bbox import processBoxes
from Detection.Yolov3.utils.utils import get_yolo_boxes


class Yolov3:
    def __init__(self):
        config_path = "Detection/Yolov3/config.json"

        with open(config_path) as config_buffer:
            self.config = json.load(config_buffer)

        ###############################
        #   Set some parameter
        ###############################
        self.net_h, self.net_w = 416, 416  # a multiple of 32, the smaller the faster
        self.obj_thresh, self.nms_thresh = 0.5, 0.3

        ###############################
        #   Load the model
        ###############################
        # os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
        self.infer_model = load_model(self.config['train']['saved_weights_name'])

    def detect(self, image):
        # predict the bounding boxes
        boxes = get_yolo_boxes(self.infer_model, [image], self.net_h, self.net_w,
                               self.config['model']['anchors'], self.obj_thresh, self.nms_thresh)[0]
        boxes = processBoxes(image, boxes, self.config['model']['labels'], self.obj_thresh)
        print(boxes)

        return boxes
