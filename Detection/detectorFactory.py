#from Detection.Detector import *
import sys
import utils.hyper_parameters as hp

class DetectorFactory():
    @staticmethod
    def getDetector(name):
        o_detector = None
        if name == "gyv3":
            from Detection.Detector import Detector
            o_detector = Detector()
        elif name == "yv3":
            from Detection.Yolov3.Yolov3 import Yolov3
            o_detector = Yolov3()

        elif name == "yolov4":
            from Detection.Detector_yolov4 import Detector_v4
            o_detector = Detector_v4()
        
        elif name == "ScaledYoloV4":
            from Detection.yolo_v4_detect import yolov4
            o_detector = yolov4(hp.weightFile, hp.img_size)

        elif name == "yolov5":
            import torch
            from Detection.Yolov5.detect_yolov5 import Yolov5
            hp.detectorType = "det"
            with torch.no_grad():
                if hp.front:
                    o_detector = Yolov5(weights = hp.modelPathFront, path="Detection/Yolov5/",imageSize=hp.img_sizeFront)
                else:
                    o_detector = Yolov5(weights = hp.modelPath, path="Detection/Yolov5/",imageSize=hp.img_size)
                    
            
        return o_detector
"""
if __name__ == "__main__":
    yolov4 = DetectorFactory()
    yolov4.getDetector("ScaledYoloV4")
"""
