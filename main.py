import os
import pdb
import sys
import time
from datetime import datetime
from parseArgs import parseArgs
from Detection.detectorFactory import *
from Tracker_class_dlib import DetectionTracker
from utils.csv_writer import CsvWriter
from utils.dataLoader import DataLoader
from utils.utils import *
from offline_backtracking import offline_bt_ver2 as pp



class Main:
    def __init__(self):
        self.data = DataLoader()
        self.o_csvWriter = CsvWriter(self.data.getDetectorName(), self.data.getHyperparamName())
        self.o_detector = DetectorFactory.getDetector(hp.detector)
        self.o_tracker = DetectionTracker()
        

    def Run(self):
        count = 0
        while True:
            # Get frame to process
            frame, isAvailable, frameInfo, detResult = self.data.getImage()
            count += 1
            # print("\nFrame",count)
            if hp.debugger:
                pdb.set_trace()

            # Conditions to either continue or stop
            if isAvailable == -1 or frame is None:
                break
            if hp.skipFrame != 0 and count % (hp.skipFrame + 1) != 0:
                continue
            if hp.maxFrames is not None and count > hp.maxFrames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detector
            timestamp = str(datetime.now())
            start = time.time()

            if type(detResult) == type(None):
                objectInfo = self.o_detector.detect(frame)
            else:
                objectInfo = detResult

            # print("detection done")
            objectInfo_conv = detectionToDetDict(objectInfo,frame)
            # for obj in objectInfo_conv:
            #     print("\n detection",obj)
            end = time.time()
            detectorTime = end - start

            # Tracker
            start = time.time()

            objects = self.o_tracker.update(objectInfo_conv,rgb)

            # for obj in objects:
            #     print("\n update",obj,objects[obj].state,objects[obj].box,objects[obj].stateCount)
            frame,anonymiseObjects = self.o_tracker.updateAnonymize(frame)
            self.o_tracker.orientation(frame)
            # blurQuotients = self.o_tracker.updateBlurQuotient(frame)

            end = time.time()
            trackerTime = end - start
            
                # Store the videos of inference
            if hp.storeDetection:
                # frame = visualizeDetections(frame, objects)
                # self.data.write(frame, fileName=frameInfo)

                # Store the inference result in csv format
                self.o_csvWriter.writeCsv(count, frameInfo, timestamp, objects,anonymiseObjects)
            
#            sys.stdout.write('\r>> Processing %d/%d images' % (count, self.data.countFrames))
#            sys.stdout.flush()
        self.o_csvWriter.csvFile.close()
        self.data.close()
        return count

    def postProcess(self,framesCount):
        # print("Kindly note this mode is only available in offline processing.\
        #  For online, kindly adjust the values for active & active prediction states.")
        self.ps = pp.postProcessing(framesCount)
        # self.ps = pp.postProcessing()
        self.ps.eliminateStates()



if __name__ == "__main__":
    parseArgs()
    st = time.time()
    runner = Main()
    count = runner.Run()
 #   print("\nExecution Time:",time.time()-st)
 #   print("\n Post-Processing")
    runner.postProcess(count-1) # total frames are lesser than break condition
    print("\nExecution Time:",time.time()-st)
    
