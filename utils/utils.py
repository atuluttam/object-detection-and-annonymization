
import random

import cv2
import numpy as np

from utils.hyper_parameters import frame_height
from utils.hyper_parameters import storedDet
import utils.hyper_parameters as hp

def detectionToDetDict(dets,frame):
    if type(dets)==list:
        dets = np.array(dets,dtype=object)

    detsList = []
    h,w,_ = frame.shape
    if storedDet == None:
        for i in range(len(dets)):
            try:
                class_name = dets[i][0].decode('ascii')
            except:
                class_name = dets[i][0]
            confidence = dets[i][1]
            bbox = dets[i][2]
            a = max(0,bbox[0] - bbox[2] / 2)
            b = max(0,bbox[1] - bbox[3] / 2) 
            c = min(bbox[0] + bbox[2] / 2 , w)
            d = min(bbox[1] + bbox[3] / 2 , h)
            detsDict = {"class":class_name, "box" : [a, b, c, d], "conf":confidence}
            detsList.append(detsDict)
        return detsList
    else:
        for i in range(len(dets)):
            try:
                class_name = dets[i][0].decode('ascii')
            except:
                class_name = dets[i][0]
                confidence = 0
                bbox = [dets[i][1],dets[i][2],dets[i][3],dets[i][4]]
                a = max(0,bbox[0] - bbox[2] / 2)
                b = max(0,bbox[1] - bbox[3] / 2) 
                c = min(bbox[0] + bbox[2] / 2 , w)
                d = min(bbox[1] + bbox[3] / 2 , h)

                detsDict = {"class":class_name, "box" : [a, b, c, d], "conf":confidence}
                detsList.append(detsDict)
        return detsList
        
    

def detectionToTracker(dets, frameNo, removeframeList):
    dets_list = []
    if frameNo in removeframeList:
        dets = [dets[random.randrange(len(dets))]]
    dets_np = np.array(dets)
    for i in range(len(dets_np)):
        try:
            class_name = dets[i][0].decode('ascii')
        except:
            class_name = dets[i][0]
        confidence = dets[i][1]
        bbox = dets[i][2]
        a = bbox[0] - bbox[2] / 2
        b = bbox[1] - bbox[3] / 2
        c = bbox[0] + bbox[2] / 2
        d = bbox[1] + bbox[3] / 2
        dets_list.append((class_name, a, b, c, d, confidence))
    return dets_list


states = {"Pretrack": (255, 0, 0), "Active": (0, 255, 0), "Inactive": (0, 0, 255), "Active_Prediction": (255, 255, 0)}

def visualizeDetections(frame,dets):
    if len(dets) == 0:
        return frame
    for objid,obj in dets.items():
        box = int(obj.box[0]) ,int(obj.box[1]) , int(obj.box[2]), int(obj.box[3])
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
    (0, 255, 0), 2)
        frame = cv2.putText(frame, str(obj.objectID) +str(obj.objClass), (int(box[0]+20), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 1, cv2.LINE_AA)
    return frame

def vis_detections(im, dets, mode, frameNo, thresh=0.7, trackWidth=2, detectWidth=1):
    """Draw detected bounding boxes."""
    dets_np = np.array(dets)
    if len(dets_np) == 0:
        return im

    Height, Width = im.shape[:2]
    if (mode == 't'):
        instruction(im, frameNo)

    for i in range(len(dets_np)):
        if (mode == 't'):
            class_name = dets[i][2] + str(dets[i][0])
            a = dets[i][3]
            b = dets[i][4]
            c = dets[i][5]
            d = dets[i][6]
            drawBoxW = trackWidth

            rectangleColor = states[dets[i][8]]
            if dets[i][8] == "Inactive":
                continue
        elif (mode == 'd'):
            try:
                class_name = dets[i][0].decode('ascii')
            except:
                # class_name = dets[i][0] + str(dets[i][1]).split(".")[1][:3]
                class_name = dets[i][0]
            a = dets[i][1]
            b = dets[i][2]
            c = dets[i][3]
            d = dets[i][4]
            drawBoxW = detectWidth

            rectangleColor = (0, 0, 0)
        else:
            break
        bbox_width = int(float(c)) - int(float(a))
        bbox_height = int(float(d)) - int(float(b))
        text_width = (bbox_width / Width)
        if (text_width < 0.4):
            text_width = 0.5
        elif (text_width > 0.8):
            text_width = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX

        im = cv2.rectangle(im, (int(a), int(b)), (int(c), int(d)), rectangleColor, drawBoxW)
        text_shape = cv2.getTextSize(class_name, font, fontScale=text_width, thickness=1)[0]
        x1 = int(a - 1)
        y1 = int(d - text_shape[1] - 1)
        x2 = int(a + text_shape[0] + 1)
        y2 = int(d + 1)

        # box_coords = ((a-text_height-1, b-1), (a+1,b+text_width+1))
        # cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
        im = cv2.putText(im, class_name, (int(a), int(d)), cv2.FONT_HERSHEY_SIMPLEX,
                         text_width, (0, 0, 255), 1, cv2.LINE_AA)

    return im


def instruction(frame, frameNo):
    # Write on image Header message
    msg = "Tracker States:"
    yVal = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_shape = cv2.getTextSize(msg, font, fontScale=0.5, thickness=1)[0]
    cv2.putText(frame, msg, (3, yVal), font,
                0.5, (255, 0, 0), 1, cv2.LINE_AA)
    yVal += text_shape[1]

    # Write on image the tracker states and its corresponding color
    for state in states:
        text_shape = cv2.getTextSize(state, font, fontScale=0.5, thickness=1)[0]
        cv2.putText(frame, state, (3, yVal), font,
                    0.5, states[state], 1, cv2.LINE_AA)
        yVal += text_shape[1]

    # Write on image the frame number
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_shape = cv2.getTextSize(str(frameNo), font, fontScale=0.5, thickness=1)[0]
    cv2.putText(frame, str(frameNo), (3, frame_height - text_shape[1]), font,
                0.5, (255, 0, 0), 1, cv2.LINE_AA)


def imageBlur(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
    if kW < 2 or kH < 2:
        return image
    return cv2.GaussianBlur(image, (kW, kH), 0)

def imageBlur(image):
    w, h = (5, 5)
    height, width = image.shape[:2]
    temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return output