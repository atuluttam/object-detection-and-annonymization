import numpy as np
from utils.utils import imageBlur
import cv2
import dlib
from tracking.association import Association
from collections import OrderedDict

class Annonymise:
    def __init__(self):
        self.nextObjectID = 1
        self.objDict = OrderedDict()
        self.tracker = OrderedDict()
        self.trackBox = OrderedDict()
        self.associate = Association()
        self.matchingIouThresh = 0.01  
        self.faultyTrackerIoU = 0.7
        self.minPeakToSideLobeRatio = 5
        self.faceBox = None

    def register(self,obj,frameRGB):
        self.objDict[self.nextObjectID] = obj
        self.tracker[self.nextObjectID] = dlib.correlation_tracker()
        self.trackBox[self.nextObjectID] = None
        self.startTrack(self.tracker[self.nextObjectID],frameRGB,obj)
        self.nextObjectID += 1

    def deregister(self,objID):
        del self.objDict[objID]
        del self.tracker[objID]
        del self.trackBox[objID]

    def annonymizeFace(self,frame,state,box):  
        # if (state != "Inactive" or state != "Delete"):
        wOrig = box[2] - box[0]
        h = box[3] - box[1]
        ar = round(wOrig/h,2)
        # print("ar: ",ar)
        if ar > 0.6:
            w = max(h*0.3, wOrig)
            face_xmin = max(0,int(box[0] + wOrig/2 - w/2))
            face_ymin = max(0,int(box[1]))
            face_xmax = min(int(face_xmin + w),int(box[2]))
            face_ymax = min(int(face_ymin + h*min(ar,1)*0.9),int(box[3]))
        else:
            w = max(h*0.3, wOrig*0.8)
            face_xmin = max(0,int(box[0] + wOrig/2 - w/2))
            face_ymin = max(0,int(box[1]))
            face_xmax = min(int(face_xmin + w),int(box[2]))
            face_ymax = min(int(face_ymin + h*0.3),int(box[3]))
            
        self.faceBox = [face_xmin, face_ymin,face_xmax,face_ymax]
        # cv2.rectangle(frame, (face_xmin, face_ymin), (face_xmax, face_ymax),(255,0, 0), 2)
        faceImg = frame[face_ymin:face_ymax,face_xmin:face_xmax]
        if 0 in faceImg.shape:
            return frame
        faceImg = imageBlur(faceImg)
        # faceImg = imageBlur(faceImg)
        frame[face_ymin:face_ymax,face_xmin:face_xmax] = faceImg                    
        return frame

    def startTrack(self,dlibTracker,frameRGB,box):
        box = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        dlibTracker.start_track(frameRGB,box)
        
    def predictCoords(self,objID,frameRGB,Vehiclebox,frame):
        dlibTracker = self.tracker[objID]
        trackingConfidence = dlibTracker.update(frameRGB)
        pos = dlibTracker.get_position()
        startX = max(0,int(pos.left()))
        startY = max(0,int(pos.top()))
        endX = min(int(pos.right()),frameRGB.shape[1])
        endY = min(int(pos.bottom()),frameRGB.shape[0])
        # cv2.rectangle(frame, (startX,startY), (endX,endY),(255,0, 0), 3)
        if Vehiclebox[0]<=startX and Vehiclebox[1]<=startY and Vehiclebox[2]>=endX and Vehiclebox[3]>=endY:
            self.trackBox[objID] = (startX, startY, endX, endY)
        else:
            trackingConfidence=0
        return trackingConfidence
        
    def trackerKeepOrDelete(self,objID,frameRGB,Vehiclebox,frame):
        conf = self.predictCoords(objID,frameRGB,Vehiclebox,frame)
        if conf >= self.minPeakToSideLobeRatio:
            self.objDict[objID] = self.trackBox[objID]
        else:
            self.deregister(objID)
            
    def faultyTracker(self,frameRGB,a,b,objID):
        intrs = (np.minimum(a[2],b[2]) - np.maximum(a[0],b[0])) * (np.minimum(a[3],b[3]) - np.maximum(a[1],b[1]))
        if intrs < 0:
            iou = 0
        else:
            iou = intrs / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intrs)
        if (iou < self.faultyTrackerIoU):
            self.trackBox[objID] = None
            self.startTrack(self.tracker[objID],frameRGB,self.objDict[objID])

