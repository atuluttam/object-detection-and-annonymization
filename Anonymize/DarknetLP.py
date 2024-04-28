from Detection.Detector import Detector
from utils.utils import imageBlur
class DarknetLP:
    def __init__(self):
        cfg = {
            'modelPath': './Anonymize/DarknetLPcfg/libdarknet.so',
            'configurationFile': b'./Anonymize/DarknetLPcfg/darknet-yolov3.cfg',
            'weightFile': b"./Anonymize/DarknetLPcfg/lapi.weights",
            'dataFile': b"./Anonymize/DarknetLPcfg/NumberPlate.data"
        }
        self.detector = Detector(cfg)

    def detect(self,image,frameNo, baseXmin, baseYmin):
        darkDetect = self.detector.detect(image, thresh=0.1)
        detections = []
        for eachDet in darkDetect:
            bbox = eachDet[2]

            xmin = int(bbox[0] - bbox[2] / 2)
            ymin = int(bbox[1] - bbox[3] / 2)
            xmax = int(bbox[0] + bbox[2] / 2)
            ymax = int(bbox[1] + bbox[3] / 2)

            LP = image[ymin:ymax, xmin:xmax]
            LP = imageBlur(LP)
            image[ymin:ymax, xmin:xmax] = LP

            xmin = int(baseXmin + xmin)
            ymin = int(baseYmin + ymin)
            xmax = int(baseXmin + xmax)
            ymax = int(baseYmin + ymax)

            frmt = []
            frmt.append(0)
            frmt.append(frameNo)
            frmt.append("LP")
            frmt.append(xmin)
            frmt.append(ymin)
            frmt.append(xmax)
            frmt.append(ymax)
            frmt.append(eachDet[1])


            detections.append(frmt)
        return detections, image

