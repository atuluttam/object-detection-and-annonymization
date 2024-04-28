import numpy as np
import dlib
from collections import Counter
import utils.hyper_parameters as hp

class StateMachineTracker:
    def __init__(self,objID,obj,frameRGB):
        self.objectID = objID
        self.stateCount = 0
        self.state = "PreTrack"
        self.objConf = obj["conf"]
        self.preTrackThresh = hp.pretrackToActiveThresh
        self.box = obj["box"]
        self.detBox = obj["box"]
        self.trackBox = None
        self.objClassCountThresh = 5
        # self.pedCategories = ['person']
        self.activePredThresh = hp.predictionToInactiveThresh
        self.inactiveThresh = hp.inactiveToDeleteThresh
        self.objClassCount = [obj["class"]]  
        self.faultyTrackerIoU = 0.8  
        # self.objClass = Counter(self.objClassCount).most_common(1)[0][0]
        self.objClass = obj["class"]
        self.preTrackState(self.objConf,self.detBox,self.objClass,frameRGB)
        self.minSLR = 8
        self.orientation = None
        self.orientationConf = 0

    def preTrackState(self,conf,box,objClass,frameRGB):
        self.stateCount += 1
        self.detBox = box
        self.objConf = conf
        self.objClass = objClass
        # self.refreshClassCounter()
        # self.objClassCount.append(objClass)
        # self.objClass = Counter(self.objClassCount).most_common(1)[0][0]
        if self.stateCount >= self.preTrackThresh:
            self.state = "Active"
            self.tracker = dlib.correlation_tracker()
            self.startTrack(frameRGB)
        self.box = self.detBox

    def activeState(self,box,objClass,frameRGB):
        self.state = "Active"
        self.detBox = box
        self.objClass = objClass
        # self.refreshClassCounter()
        # self.objClassCount.append(objClass)
        # self.objClass = Counter(self.objClassCount).most_common(1)[0][0]
        self.stateCount = self.preTrackThresh
        self.predictCoords(frameRGB)
        self.box = self.detBox
        self.faultyTracker(frameRGB)

    def activePredictionState(self,frameRGB):
        self.state = "ActivePrediction"
        conf = self.predictCoords(frameRGB)
        self.stateCount += 1
        if self.stateCount > self.activePredThresh or conf < self.minSLR:
            self.state = "Inactive"
        self.box = self.trackBox
            
    def inactiveState(self):
        self.stateCount += 1
        if self.stateCount > self.inactiveThresh:
            self.state = "Delete"
        self.box = self.trackBox
        
    def startTrack(self,frameRGB):
        startX, startY, endX, endY = int(self.detBox[0]), int(self.detBox[1]), int(self.detBox[2]), int(self.detBox[3])
        box = dlib.rectangle(startX,startY,endX,endY)
        self.tracker.start_track(frameRGB,box)
        
    def predictCoords(self,frameRGB):
        conf = self.tracker.update(frameRGB)
        pos = self.tracker.get_position()
        startX = max(0,int(pos.left()))
        startY = max(0,int(pos.top()))
        endX = min(int(pos.right()),frameRGB.shape[1])
        endY = min(int(pos.bottom()), frameRGB.shape[0])
        self.trackBox = (startX, startY, endX, endY)
        return conf

    def faultyTracker(self,frameRGB):
        a = self.detBox
        b = self.trackBox
        intrs = (np.minimum(a[2],b[2]) - np.maximum(a[0],b[0])) * (np.minimum(a[3],b[3]) - np.maximum(a[1],b[1]))
        if intrs < 0:
            iou = 0
        else:
            iou = intrs / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intrs)
        if iou < self.faultyTrackerIoU:
            self.startTrack(frameRGB)

    def refreshClassCounter(self):
        if len(self.objClassCount)>= self.objClassCountThresh:
            self.objClassCount = []