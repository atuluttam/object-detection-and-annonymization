import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import  numpy as np
import sys
# from Detection.Yolov5.LPDetector.models.yolo import Model
from utils import hyper_parameters as hp
from Detection.Yolov5.models.experimental import attempt_load
from Detection.Yolov5.utils.general import check_img_size,non_max_suppression,scale_coords
from Detection.Yolov5.utils.augmentations import letterbox
from Detection.Yolov5.utils.torch_utils import select_device

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class Yolov5:
    def __init__(self, weights, path, imageSize=640,detectType=None,stateDict=None,configFile=None,names=None):
        self.path = path
        with add_path(path):
            self.view_img = False
            self.save_txt = False
            self.imgsz = imageSize
            self.device = "cpu"
            self.augment = False
            self.conf_thres = 0.4
            self.iou_thres = 0.5
            self.classes = 0
            self.agnostic_nms = True

            self.device = select_device(self.device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            # if detectType == "LP":
            #     self.model = Model(configFile).to(self.device)
            #     self.model.load_state_dict(torch.load(stateDict, map_location=self.device))
            #     self.names = names
            # else:
            if True:
            # Load model
                # print("Loading model:",weights,self.imgsz)
                self.model = attempt_load(weights, map_location=self.device)
                self.stride = int(self.model.stride.max())
                self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                
            # self.model.to(self.device).eval() 
   
            if self.half:
                self.model.half()  # to FP16
            # check img_size

            # imgsz = check_img_size(self.imgsz, s=self.stride)
            # if self.device.type != 'cpu':
            #     self.model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

            # Get names and colors
            

            # img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
            # _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
    

    def detect(self, image):
        with add_path(self.path):
            # for path, img, im0s, vid_cap in dataset:
            img = letterbox(image,self.imgsz,stride=int(self.model.stride.max()),auto=True)[0]
            img =  img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            pred1 = self.model(img, augment=self.augment,visualize=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred1, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms,max_det=1000)

            result = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    # det = nms(det, self.iou_thres)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = self.names[int(cls)]
                        if label in hp.acceptableCategories:
                            coord = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                            x = (coord[0] + coord[2])/2
                            y = (coord[1] + coord[3])/2
                            width = coord[2] - coord[0]
                            height = coord[3] - coord[1]
                            val = (label, conf.tolist(), (x, y, width, height))
                            result.append(val)
                    # image = cv2.rectangle(image, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0,255,0), 1)
        # cv2.imwrite("inference/result.png", image)
        return result

  

if __name__ == '__main__':
    source = '/media/kitti-tracking/image_02/data'
    images = [image for image in os.listdir(source) if
                   ("jpg" in image or "png" in image)]
    img = cv2.imread(os.path.join(source, images[1]))

    oDetect = Yolov5(weights= "LPDetector/best_320.pt",
                                   imageSize=320, path="LPDetector/")
    print(oDetect.detect(img))
