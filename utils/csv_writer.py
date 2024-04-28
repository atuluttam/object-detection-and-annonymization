import csv
import platform

import utils.hyper_parameters as hp


class CsvWriter:
    def __init__(self, detFileName, hyperparamFileName):
        fieldnames = ["Sequence Number", "Frame number", "Timestamp", "Obj_ID", "Obj_X1", "Obj_X2", "Obj_Y1", "Obj_Y2", "LPBoxes",
                      "Obj_Confidence", "Obj_Classification","State_Machine","SM_count","orientation","orientationConf","blurQuotient"]
        self.csvFile = open(detFileName, 'w', newline='')
        self.writer = csv.DictWriter(self.csvFile, fieldnames=fieldnames)
        self.writer.writeheader()
        # self.writeHyperparameters(hyperparamFileName)

    def __del__(self):
        self.csvFile.close()


    def writeHyperparameters(self, hyperparamFileName):
        fieldnames = ["Association IOU", "Pre-Track Threshold", "Maximum Frames Threshold",
                      "Maximum Non-detetection Threshold", "Frame Width", "Frame Height", "fps",
                      "Input Path", "Run Name", "Input Type", "Skip Frame", "Remove Detection List"]

        fieldInfo = [hp.association_iou, hp.Pre_track_threshold, hp.max_frames_threshold, hp.max_non_det_threshold,
                     hp.frame_width, hp.frame_height, hp.fps, hp.input_path, hp.runName, hp.input_type,
                     hp.skipFrame, hp.removeDetectionList]

        with open(hyperparamFileName, 'w', newline='') as file:
            # writer = csv.writer(file)
            # writer.writerow(fieldnames)
            for name, info in zip(fieldnames, fieldInfo):
                file.write(name + ": " + str(info) + "\n")
            uname = platform.uname()
            file.write("System: " + str(uname.system) + "\n")
            file.write("Machine: " + str(uname.machine) + "\n")

            if str(uname.system) == "Linux":
                file.write("CPU Info: \n")
                with open("/proc/cpuinfo", "r")  as f:
                    info = f.readlines()

                cpuinfo = [x.strip().split(":")[1] for x in info if "model name" in x]
                file.writelines(cpuinfo[0])
            file.write("UKF: " + str(hp.UKF) + "\n")
            file.write("FPDetectionRatio: " + str(hp.FPDetectionRatio) + "\n")
            file.write("FPOnObject: " + str(hp.FPOnObject) + "\n")

    def writeCsv(self, seqNumber, frameNumber, timeStamp, dets,anonymiseObjects,blurQuotients=None):
        for objid,obj in dets.items():
            if obj.objClass != "person":
                self.writer.writerow({'Sequence Number': seqNumber,
                                        'Frame number': frameNumber,
                                        'Timestamp': timeStamp,
                                        'Obj_ID': int(obj.objectID),
                                        'Obj_X1': int(obj.box[0]),
                                        'Obj_X2': int(obj.box[2]),
                                        'Obj_Y1': int(obj.box[1]),
                                        'Obj_Y2': int(obj.box[3]),
                                        'LPBoxes':list(anonymiseObjects[objid].objDict.values()),
                                        'Obj_Confidence': obj.objConf,
                                        'Obj_Classification': obj.objClass,
                                        'State_Machine': obj.state,
                                        'SM_count': obj.stateCount,
                                        'orientationConf' : obj.orientationConf,
                                        'orientation' : obj.orientation,
                                        'blurQuotient':0}
                                        )

            else:
                self.writer.writerow({'Sequence Number': seqNumber,
                        'Frame number': frameNumber,
                        'Timestamp': timeStamp,
                        'Obj_ID': int(obj.objectID),
                        'Obj_X1': int(obj.box[0]),
                        'Obj_X2': int(obj.box[2]),
                        'Obj_Y1': int(obj.box[1]),
                        'Obj_Y2': int(obj.box[3]),
                        'LPBoxes': [anonymiseObjects[objid].faceBox],
                        'Obj_Confidence': obj.objConf,
                        'Obj_Classification': obj.objClass,
                        'State_Machine': obj.state,
                        'SM_count': obj.stateCount,
                        'orientationConf' : obj.orientationConf,
                        'orientation' : obj.orientation,
                        'blurQuotient': 0}
                        )

            # elif anonymiseObjects[objid].LPBox!=None:
            #     self.writer.writerow({'Sequence Number': seqNumber,
            #             'Frame number': frameNumber,
            #             'Timestamp': timeStamp,
            #             'Obj_ID': int(obj.objectID),
            #             'Obj_X1': int(obj.box[0]),
            #             'Obj_X2': int(obj.box[2]),
            #             'Obj_Y1': int(obj.box[1]),
            #             'Obj_Y2': int(obj.box[3]),
            #             'Ano_X1':anonymiseObjects[objid].LPBox[0],
            #             'Ano_X2':anonymiseObjects[objid].LPBox[2],
            #             'Ano_Y1':anonymiseObjects[objid].LPBox[1],
            #             'Ano_Y2':anonymiseObjects[objid].LPBox[3],
            #             'Obj_Confidence': obj.objConf,
            #             'Obj_Classification': obj.objClass,
            #             'State_Machine': obj.state,
            #             'SM_count': obj.stateCount})
                        
            # elif anonymiseObjects[objid].faceBox!=None:
            #     self.writer.writerow({'Sequence Number': seqNumber,
            #             'Frame number': frameNumber,
            #             'Timestamp': timeStamp,
            #             'Obj_ID': int(obj.objectID),
            #             'Obj_X1': int(obj.box[0]),
            #             'Obj_X2': int(obj.box[2]),
            #             'Obj_Y1': int(obj.box[1]),
            #             'Obj_Y2': int(obj.box[3]),
            #             'Ano_X1': anonymiseObjects[objid].faceBox[0],
            #             'Ano_X2': anonymiseObjects[objid].faceBox[2],
            #             'Ano_Y1': anonymiseObjects[objid].faceBox[1],
            #             'Ano_Y2': anonymiseObjects[objid].faceBox[3],
            #             'Obj_Confidence': obj.objConf,
            #             'Obj_Classification': obj.objClass,
            #             'State_Machine': obj.state,
            #             'SM_count': obj.stateCount})


            # else:
            #     print("csv not defined")
            #     pass
 

