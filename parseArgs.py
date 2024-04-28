import argparse
import utils.hyper_parameters as hp
from pathlib import Path


def parseArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', action='store', dest='inputPath', default=None, help="Enter the input path")
    ap.add_argument('-o', action="store", dest="outputPath", default=None, help="Enter the output path")
    ap.add_argument('-type', action='store', dest='inputType', default=r"image", help="Input type video, image")
    ap.add_argument('-OPtype', action='store', dest='outputType', default=r"image", help="Input type video, image")
    ap.add_argument('-nostore', dest='noVideo', action='store_false', help="Set the arg to disable video output")
    ap.add_argument("-prefix",dest="prefix",action="store",help="add prefix to folder name(addingsprint & dataset batch")
    args = ap.parse_args()

    hp.input_path = args.inputPath
    hp.outputPath = args.outputPath
    hp.input_type = args.inputType
    hp.OPType = args.outputType
    hp.storeDetection = args.noVideo
    if args.inputType == "image":
        hp.runName = str(args.prefix) + Path(hp.input_path).name
    else:
        hp.runName = str(args.prefix) + (Path(hp.input_path).name).split(".")[0]
    hp.front = True
    if "front" in hp.runName.lower():
        hp.front=True
    else:
        hp.front=False
    # if ("left" in hp.runName.lower()) or ("right" in hp.runName.lower()):
    #     hp.pretrackToActiveThresh = 1
    #     hp.predictionToInactiveThresh = 1
    #     hp.inactiveToDeleteThresh = 1



