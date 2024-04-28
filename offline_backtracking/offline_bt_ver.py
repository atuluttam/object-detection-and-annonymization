import sys
sys.path.append("..")
import pandas as pd
from utils import hyper_parameters as hp
import cv2
import os
import time
from tracking.association import Association
from utils.dataLoader import DataLoader
import numpy as np
import json
import operator
import parseArgs
from utils.utils import *
from utils.utils import imageBlur

class postProcessing():
    def __init__(self):
        self.saveDir = os.path.join(hp.outputPath, hp.runName)
        self.processedFilePath = os.path.join(self.saveDir, hp.runName + ".csv")
        self.iou = Association()
        self.data = DataLoader()
        self.roi=[520,450,520+1492,450+412] #x1,y1,x2,y2

    # Function return a dictionary that contains the track of frames on which object is found
    # sample 1 : [1,45] -> object with ID 1 was discovered on frame 1 & tracked till frame 45
    def saveTrajectoriesInformation(self,df,save=False):
        trajectoryDict = {}
        groups = df.groupby("Object_ID")
        for i in groups:
            trajectoryDict[i[0]] = [i[1]["Frame_Number"].min(),i[1]["Frame_Number"].max()]
        if save:
            saveJson = {}
            for i in trajectoryDict:
                saveJson[i] = str(trajectoryDict[i])
            with open(os.path.join(self.saveDir, hp.runName + "objectTrajectories.json"), 'w', encoding='utf-8') as f:
                json.dump(saveJson, f,indent=4)
                del saveJson
        return trajectoryDict
                
        
    # Function to reorder the object ID in consecutive order after eliminating unnecessary object ID's
    def redefineID(self):
        trajectoryDict = self.saveTrajectoriesInformation(self.result)
        sortedIDs = sorted(trajectoryDict.items(), key=operator.itemgetter(1))
        uniqueIdDict = {}

        for ind,i in enumerate(sortedIDs):
            uniqueIdDict[i[0]] = ind+1

        objNewId = []
        for i in self.result["Object_ID"]:
            objNewId.append(uniqueIdDict[i])
        return objNewId

    # Function crops the bboxes to get fit under given ROI region
    def ROI(self,df):
        originalDF = df.copy()
        def Roix1(x):
            return(max(self.roi[0],x))
        def Roix2(x):
            return(min(self.roi[2],x))
        def Roiy1(x):
            return(max(self.roi[1],x))
        def Roiy2(x):
            return(min(self.roi[3],x))
        df["X1"]=df["X1"].apply(Roix1)
        df["Y1"]=df["Y1"].apply(Roiy1)
        df["X2"]=df["X2"].apply(Roix2)
        df["Y2"]=df["Y2"].apply(Roiy2)
        df["w"] = df["X2"] - df["X1"]
        df["h"] = df["Y2"] - df["Y1"]
        df["area"] = df["w"]*df["h"]
        df = df[df["area"]>0]
        df.drop(['area', 'w','h'], axis=1,inplace=True)
        return df
   
# main function which eliminates the objects from active prediction state to avoid some FP
# It counts last active state after which object goes to active_prediction & inactive state
# Taking the last active count, it removes all entries after that, thereby helping in decreasing
# False Positives
    def eliminateStates(self):
        data = pd.read_csv(self.processedFilePath)
        df = pd.DataFrame(columns=list(data.columns))
        dataFrames = [df]
        groups = data.groupby("Obj_ID")
        for i in groups.groups:
            group = groups.get_group(i)
            subGroup = group.groupby("State_Machine")
            last_index = (subGroup.get_group("Active").index)[-1]
            if len(group.loc[:last_index,:]) > 1:
                dataFrames.append(group.loc[:last_index+1,:])
        self.result = pd.concat(dataFrames)
        self.result = self.result[["Sequence Number","Frame number","Obj_Classification","Obj_ID","Obj_X1","Obj_Y1","Obj_X2","Obj_Y2","Obj_Confidence","LPBoxes"]]
        self.result.columns=["Frame_Number","Image_Name","Object_Class","Object_ID","X1","Y1","X2","Y2","Confidence","LPBoxes"]
        acceptableCategories = ['person', 'car', 'motorcycle', 'bus', 'train', 'truck']
        self.result = self.result[self.result["Object_Class"].isin(acceptableCategories)]
        print(self.result.shape)
        # self.result = self.ROI(self.result)
        # print("shape after ROI clipping:",self.result.shape)
        objNewId = self.redefineID()
        self.result["Object_ID"] = objNewId
        self.LPBoxesError(self.result)
        self.saveTrajectoriesInformation(self.result,save=True)
        result = self.result.to_json(orient="records")
        parsed = json.loads(result)
        with open(os.path.join(self.saveDir, hp.runName + "_processed.json"), 'w') as json_file:
            json.dump(parsed, json_file)
        self.result.to_csv(os.path.join(self.saveDir, hp.runName + "_processed.csv"),index=False)
    
                    
    def visualizeDetections(self,frame,obj):
        # boxes = obj[-1]
        # if boxes=='[]':
        #     pass
        # else:
        #     for box in boxes:
        #         lpImg = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        #         lpImg = imageBlur(lpImg)
        #         frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = lpImg

        box = int(obj[2]) ,int(obj[4]) , int(obj[3]), int(obj[5])
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),(0, 255, 0), 2)
        text = str(obj[1]) # + str(obj[6])
        frame = cv2.putText(frame, text , (int(box[0]+20), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 0, 255), 1, cv2.LINE_AA)
        return frame


    def visualizeFinalOutput(self):
        if hp.input_type == "video":
            self.hVideo = cv2.VideoCapture(hp.input_path)
            hp.frame_width = int(self.hVideo.get(3))
            hp.frame_height = int(self.hVideo.get(4))
            hp.fps = int(self.hVideo.get(5))
            self.countFrames = int(self.hVideo.get(cv2.CAP_PROP_FRAME_COUNT))

        elif hp.input_type == "image":
            self.images = [image for image in os.listdir(hp.input_path) if
                           ("jpg" in image or "png" in image)]
            self.images = sorted(self.images, key=lambda timestamp: timestamp)
            img = cv2.imread(os.path.join(hp.input_path, self.images[0]))
            hp.frame_height, hp.frame_width = img.shape[:2]
            self.countFrames = len(self.images)

        if hp.storeDetection:
            if hp.OPType == 'video':
                videoName = os.path.join(self.saveDir, hp.runName + "_processed.avi")
                self.out = cv2.VideoWriter(videoName,
                                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), hp.fps,
                                        (hp.frame_width, hp.frame_height))

        result_groups = self.result.groupby("Image_Name")
        count = 0
        while count<self.countFrames:
            if hp.input_type == "image":
                frame = cv2.imread(os.path.join(hp.input_path,self.images[count]))
                image_name = self.images[count]
            elif hp.input_type == "video":
                image_name = str(count+1).zfill(5) + ".jpg"
                _,frame = self.hVideo.read()
            if image_name in list(result_groups.groups.keys()): 
                values = result_groups.get_group(image_name)
                filterValues = values[["Image_Name","Object_ID","X1","X2","Y1","Y2","Object_Class","LPBoxes"]].iloc[:,:].values
                for val in filterValues:
                    self.visualizeDetections(frame,val)
            if hp.OPType == 'video':
                self.out.write(frame)
            else:
                cv2.imwrite(os.path.join(self.saveDir,image_name),frame)
            count+=1
        if hp.OPType == 'video':
            self.out.release()

if __name__ == "__main__":
    ps = postProcessing()
    ps.eliminateStates()
    ps.visualizeFinalOutput()


            

    