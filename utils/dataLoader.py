import csv
import os
import random
import time

import cv2
import utils.hyper_parameters as hp
import pandas as pd
import numpy as np
import skvideo.io

class DataLoader:
    def __init__(self):
        self.frameNumber = 0
        self.sortedFiles = None # matches filename with first frame in case of images
        self.configureWorkingDirectory()
        if hp.input_type == "video":
            if hp.inputReadingModeVideo == "opencv":
                self.hVideo = cv2.VideoCapture(hp.input_path)
                hp.frame_width = int(self.hVideo.get(3))
                hp.frame_height = int(self.hVideo.get(4))
                hp.fps = int(self.hVideo.get(5))
                self.countFrames = int(self.hVideo.get(cv2.CAP_PROP_FRAME_COUNT))
            elif hp.inputReadingModeVideo == "skvideo":
                self.hVideo = skvideo.io.vreader(hp.input_path)
                metadata = skvideo.io.ffprobe(hp.input_path)
                hp.frame_height = int(metadata["video"]["@height"])
                hp.frame_width = int(metadata["video"]["@width"])
                hp.fps = int(metadata["video"]["@r_frame_rate"].strip("''").split("/")[0])
                self.countFrames = int(metadata["video"]["@nb_frames"])

        elif hp.input_type == "image":
            self.images = [image for image in os.listdir(hp.input_path) if
                           ("jpg" in image or "png" in image)]
            self.images = sorted(self.images, key=lambda timestamp: timestamp)
            # self.images = [str(i).zfill(5) +".jpg" for i in range(10,900,30)]
            # # self.images = [ "{}.jpg".format(x) for x in range(0,len(self.images)*4,4)]
            img = cv2.imread(os.path.join(hp.input_path, self.images[0]))
            hp.frame_height, hp.frame_width = img.shape[:2]
            self.countFrames = len(self.images)

        if hp.storeDetection:
            if hp.OPType == 'video':
                self.resultPath = os.path.join(hp.outputPath, hp.runName)
                self.outVideoName = os.path.join(self.resultPath,hp.runName + "_output.avi")
            elif hp.OPType == 'image':
                self.resultPath = os.path.join(hp.outputPath, hp.runName) 
            else:
                print("Wrong O/P store type")
                exit(0)

            if not os.path.exists(self.resultPath):
                os.makedirs(self.resultPath)

        self.prepareStoredir()
        self.processDetectInput()
        self.removeDetectionInit()

    def prepareStoredir(self):
        if hp.storeDetection:
            if hp.OPType == 'video':
                self.out = cv2.VideoWriter(self.outVideoName,
                                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), hp.fps,
                                        (hp.frame_width, hp.frame_height))


    def close(self):
        if hp.OPType == 'video':
            self.out.release()
        if hp.inputReadingModeVideo == "opencv":
            if hp.input_type == "video":
                self.hVideo.release()

    def configureWorkingDirectory(self):
        self.workPath = os.path.join(hp.outputPath, hp.runName)
        if not os.path.exists(self.workPath):
            os.makedirs(self.workPath)
        else:
            print("Path already exist replacing in 5 sec")
            time.sleep(5)
        self.csvDetectorName = os.path.join(self.workPath, hp.runName + ".csv")
        self.csvHyperparamName = os.path.join(self.workPath, hp.runName + "HyperParameter.txt")



    def getImage(self):
        isAvailable = -1
        frame = None
        if hp.input_type == "video" and hp.inputReadingModeVideo == "opencv":
            if self.hVideo.isOpened():
                ret, frame = self.hVideo.read()
                if ret:
                    isAvailable = 1
                    self.frameNumber += 1
                if hp.OPType == 'image':
                    image_name = str(self.frameNumber).zfill(5) + ".jpg"
                else:
                    image_name = self.frameNumber
                return frame, isAvailable, image_name, self.getDetResult(image_name)
            else:
                print("Invalid video path")

        elif hp.input_type == "video" and hp.inputReadingModeVideo == "skvideo":
            for frame in self.hVideo:
                frame = frame.copy()
                frame = frame[:,:,::-1]
                if self.frameNumber < self.countFrames:
                    isAvailable = 1
                self.frameNumber += 1
                if hp.OPType == 'image':
                    image_name = str(self.frameNumber).zfill(5) + ".jpg"
                else:
                    image_name = self.frameNumber
                return frame, isAvailable, image_name, self.getDetResult(image_name)
            

        elif hp.input_type == "image":
            if self.frameNumber < len(self.images):
                image_name = self.images[self.frameNumber]
                image_path = os.path.join(hp.input_path, image_name)
                frame = cv2.imread(image_path)
                isAvailable = 1
                self.frameNumber += 1

                return frame, isAvailable, image_name, self.getDetResult(image_name)
        else:
            print("Invalid Input")
        return frame, isAvailable, None, None

    def write(self, frame, fileName):
        if hp.OPType == 'video':
            self.out.write(frame)
        elif hp.OPType == 'image':
            if hp.input_type == "video":
                opFileName = str(fileName)
            else:
                opFileName = fileName
            cv2.imwrite(os.path.join(self.resultPath, opFileName), frame)

    def getDetectorName(self):
        return self.csvDetectorName

    def getTrackerName(self):
        return self.csvTrackerName

    def getHyperparamName(self):
        return self.csvHyperparamName

    def processDetectInput(self):
        self.csvData = []
        if hp.storedDet is not None:
            self.csvData = pd.read_csv(hp.storedDet)
            groups =  self.csvData.groupby("filename")

    def getDetResult(self,image_name):
        if len(self.csvData) > 0 and len(self.csvData) <= self.frameNumber:
            exit(2)
        if len(self.csvData) == 0 and len(self.csvData) <= self.frameNumber:
            return None
        else:
            img_name = int((image_name.split("."))[0])
            print(img_name)
            data = self.csvData[self.csvData["filename"] == img_name].iloc[:,1:].values
            return data

    def removeDetectionInit(self):
        if hp.removeDetection != 0:
            detToRemove = int((hp.removeDetection / 100) * self.countFrames)
            hp.removeDetectionList = list(set(random.sample(range(0, self.countFrames), detToRemove)))
            print("Remove detection list")
            print(hp.removeDetectionList)
        else:
            hp.removeDetectionList = []

    def createRandomBoxes(self, count, avgWidth, avgHeight):
        boxes = []
        for i in range(count):
            className = "car"
            confidence = 0.90
            xmin = random.randrange(hp.frame_width - avgWidth)
            ymin = random.randrange(hp.frame_height - avgHeight)
            if hp.FPDetectionRatioAvgHW:
                xmax = xmin + avgWidth
                ymax = ymin + avgHeight
            else:
                xmax = random.randrange(xmin, hp.frame_width)
                ymax = random.randrange(ymin, hp.frame_height)
            xcenter = xmin + (xmax - xmin) / 2
            ycenter = ymin + (ymax - ymin) / 2
            width = (xmax - xmin)
            height = (ymax - ymin)

            boxes.append([className, confidence, [xcenter, ycenter, width, height]])
        return boxes

    def createRandomBoxesArdObj(self, detBoxes, countFP):
        newBoxes = []
        for i in range(countFP):
            iObj = random.choice(detBoxes)
            className = iObj[0]
            confidence = iObj[1]
            xcenter = iObj[2][0]
            ycenter = iObj[2][1]
            width = iObj[2][2]
            height = iObj[2][3]
            factor = 0.2
            xcenter = random.randrange(int(xcenter - width * factor), int(xcenter + width * factor))
            ycenter = random.randrange(int(ycenter - height * factor), int(ycenter + height * factor))
            width = random.randrange(int(width - width * factor), int(width + width * factor))
            height = random.randrange(int(height - height * factor), int(height + height * factor))
            newBoxes.append([className, confidence, [xcenter, ycenter, width, height]])
        return newBoxes
