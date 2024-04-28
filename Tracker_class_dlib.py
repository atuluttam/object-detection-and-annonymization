import copy
import os
import sys
import pdb
from collections import OrderedDict
import numpy as np
import utils.hyper_parameters as hp
from tracking.association import Association
from tracking.state_machine import StateMachineTracker as Tracker
from tracking.annonymise import Annonymise
from Anonymize.orientationClassifier_Pytorch import orientationClassifier
import cv2


class DetectionTracker:
    def __init__(self):
        self.nextObjectID = 1
        self.objDict = OrderedDict()
        self.anonymizeDict = OrderedDict()
        self.blurQuotientDict = OrderedDict()
        self.associate = Association()
        self.matchingIouThresh = 0.15
        self.matchingCentroidDistance = 200 #pixels
        self.orientClassifier = orientationClassifier()
        self.acceptedLPClasses = ['car','truck','bike','van','bus','motorcycle']
        self.acceptedPersonClass = ['person','pedestrian']
        self.resemblanceClass = ["car","bus","truck","van"]
        

    def register(self,obj,frameRGB):
        stateMachineObject = Tracker(self.nextObjectID,obj,frameRGB)
        anonymiseObject =  Annonymise()
        self.objDict[self.nextObjectID] = stateMachineObject
        self.anonymizeDict[self.nextObjectID] = anonymiseObject
        self.blurQuotientDict[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self,objID):
        del self.objDict[objID]
        del self.anonymizeDict[objID]
        del self.blurQuotientDict[objID]

    def mapStateToAction(self,obj,newObj,frameRGB):
        if obj.state == "PreTrack":
            obj.preTrackState(newObj["conf"],newObj["box"],newObj["class"],frameRGB)
        elif obj.state == "Active" or obj.state == "ActivePrediction" or obj.state == "Inactive" or obj.state == "Delete":
            obj.activeState(newObj["box"],newObj["class"],frameRGB)
        else:
            print("state not found",obj.state)

    # in case of missed detections or no new detections
    def existingObjectUpdate(self,targetObjects,frameRGB):
        for obj in targetObjects:
            # if obj.state == "Delete":
            #     self.deregister(obj.objectID)
            if obj.state == "PreTrack":
                obj.stateCount -= 1
                if obj.stateCount < -1:
                    obj.state = "Delete"
                    self.deregister(obj.objectID)
            elif (obj.state == "Active") or (obj.state == "ActivePrediction") :
                obj.activePredictionState(frameRGB)
            elif obj.state == "Inactive":
                obj.inactiveState()
            
    def update(self,detBoxes,frameRGB):
        registeredObjects = list(self.objDict.values())
        if len(registeredObjects) == 0 and len(detBoxes) == 0:
            return self.objDict
            # print("empty frame")

        elif len(detBoxes) == 0 and len(registeredObjects) != 0:
            # print("no new detections")
            self.existingObjectUpdate(registeredObjects,frameRGB)

        elif len(detBoxes) != 0 and len(registeredObjects) == 0:
            # print("first frame")
            for obj in detBoxes:
                self.register(obj,frameRGB)

        else:
            # print("intermediate")
            newBoxes = [obj["box"] for obj in detBoxes] 
            registeredBoxes = [obj.box for obj in registeredObjects]
            registeredIds = [obj.objectID for obj in registeredObjects]
            iouMatrix = self.associate.iouMatrix(np.array(newBoxes),np.array(registeredBoxes))
            centroidMatrix = self.associate.centroidMatrix(np.array(newBoxes),np.array(registeredBoxes))
            # print("new",np.array(newBoxes))
            # print("registered",np.array(registeredBoxes))
            # print("iou",iouMatrix)
            # print("centoid",centroidMatrix)

            usedRows = set()
            unusedRows = set()
            usedCols = set()

            rows = iouMatrix.max(axis=1).argsort()[::-1]
            cols = iouMatrix.argmax(axis=1)[rows]

            for (row, col) in zip(rows, cols):
                colID = registeredIds[col]
                if row in usedRows or colID in usedCols:
                    continue
                if (iouMatrix[row][col] >= self.matchingIouThresh):#  and (self.objDict[colID].objClass == detBoxes[row]["class"]): # and (centroidMatrix[row][col]<= self.matchingCentroidDistance):        
                    if (self.objDict[colID].objClass in self.acceptedLPClasses and detBoxes[row]["class"]  in self.acceptedPersonClass) or (self.objDict[colID].objClass in self.acceptedPersonClass and detBoxes[row]["class"]  in self.acceptedLPClasses):
                        pass
                    else:
                        self.mapStateToAction(self.objDict[colID] ,detBoxes[row],frameRGB)
                        usedRows.add(row)
                        usedCols.add(colID)
                            
                    

            unusedRows = set(range(0, iouMatrix.shape[0])).difference(usedRows)
            unusedCols = set(registeredIds) - usedCols
            # print("unused rows",unusedRows)
            # print("unused cols",unusedCols)
            for row in unusedRows:
                self.register(detBoxes[row],frameRGB)

            ununsedRegisteredObjects = [self.objDict[key] for key in unusedCols]
            self.existingObjectUpdate(ununsedRegisteredObjects,frameRGB)

            for obj in ununsedRegisteredObjects:
                if obj.state == "Delete":
                    self.deregister(obj.objectID)

        return self.objDict

    def updateAnonymize(self,frame):
        for objID,anonymizeObj in self.anonymizeDict.items():
            if self.objDict[objID].objClass in self.acceptedLPClasses:
                # if self.objDict[objID].objClass=="truck" or self.objDict[objID].objClass == "bus":
                #     frame = anonymizeObj.LPUpdate(frame,self.objDict[objID].state,self.lp_detector_640,self.objDict[objID].box,objID)
                # else:
                frame = anonymizeObj.LPUpdate(frame,self.objDict[objID].state,self.lp_detector,self.objDict[objID].box,objID)
            elif self.objDict[objID].objClass in self.acceptedPersonClass:
                frame = anonymizeObj.annonymizeFace(frame,self.objDict[objID].state,self.objDict[objID].box)
            else:
                pass
        return frame,self.anonymizeDict

    def blurQuotient(self,frame,objID):
        objImg = frame[int(self.objDict[objID].box[1]):int(self.objDict[objID].box[3]), int(self.objDict[objID].box[0]):int(self.objDict[objID].box[2])]
        gray = cv2.cvtColor(objImg, cv2.COLOR_BGR2GRAY) 
        return(round(cv2.Laplacian(gray, cv2.CV_64F).var(),2))

    def updateBlurQuotient(self,frame):
        for objID in self.objDict:
            self.blurQuotientDict[objID] = self.blurQuotient(frame,objID)
        return self.blurQuotientDict

    def orientation(self,frame):
        images = []
        objIDS = []
        maxBatchSize = 12

        for objID in self.objDict:
            box = self.objDict[objID].box
            
            image = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            if 0 in image.shape or self.objDict[objID].objClass not in self.resemblanceClass:
                continue
            lpBox = list(self.anonymizeDict[objID].objDict.values())
            if len(lpBox)>0:
                continue
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #pytorch
            images.append(image)
            objIDS.append(objID)
        import math
        batches = math.floor(len(images)/maxBatchSize) + 1

        if len(images) >= 1:
            for batch in range(batches):
                imgBatch = images[batch*maxBatchSize:(batch+1)*maxBatchSize]
                if len(imgBatch) >=1:
                    # classes,prob =  self.orientClassifier.batch_detect(imgBatch,batch_size=len(imgBatch)) #tf
                    classes,prob =  self.orientClassifier.detect(imgBatch)
                    objIDBatch = objIDS[batch*maxBatchSize:(batch+1)*maxBatchSize]
                    for ind,objID in enumerate(objIDBatch):
                        # cat,conf = self.orientClassifier.detect(image)
                        self.objDict[objID].orientationConf = prob[ind]
                        self.objDict[objID].orientation = classes[ind]

                        
        

        
                


            
        
