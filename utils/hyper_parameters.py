# --------------------------------------------------------------------------------------------------------------------

# Program - Hyperparameters
# ---------------------------------------------------------------------------------------------------------------------




detector = "yolov5"  # gyv3, yv3, yolov4, yolov5, ScaledYoloV4
    
input_path = r"None"
storedDet = None
runName = "expAnn"
input_type = "image"  # image/video
storeDetectionVideo = True
debugger = False
hasGPU = False
OPType = 'image'  
storeDetection = True
outputPath = None
acceptableCategories = ['person', 'car', 'motorcycle', 'bus', 'truck','lp',"license_plate"]
# Data information
frame_width = None
frame_height = None
fps = 10
front =  None
inputReadingModeVideo = "skvideo" #opencv/skvideo
#tracker parameters
pretrackToActiveThresh = 1
predictionToInactiveThresh = 2
inactiveToDeleteThresh = 2


# Evaluation
skipFrame = 0
removeDetectionFrame = 0
maxFrames = None
removeDetection = 0
removeDetectionList = []
FPDetectionRatio = 0
FPDetectionRatioAvgHW = False
FPOnObject = 0



# classifierModelWeights = "../Model/classifierOrient/classifierTf23.h5"
classifierModelWeights = "../Model/classifierOrient/classifierPytorch.h5"



if detector == "yolov4":
    modelPath = "../Model/yolov4/libdarknet.so"
    configurationFile = "../Model/yolov4/yolov4.cfg".encode('utf-8')
    weightFile = "../Model/yolov4/yolov4.weights".encode('utf-8')
    dataFile = b"../Model/yolov4/coco.data"

elif detector == "gyv3":
    modelPath = "../Model/gyv3/libdarknet.so"
    configurationFile = "../Model/gyv3/Gaussian_yolov3_BDD.cfg".encode('utf-8')
    weightFile = "../Model/gyv3/Gaussian_yolov3_BDD.weights".encode('utf-8')
    dataFile = b"../Model/gyv3/BDD.data"

elif detector == "ScaledYoloV4":
    # weightFile = "../Model/ScaledYOLOv4/yolov4-p7.pt"
    weightFile = "../Model/ScaledYOLOv4/yolov4-p6_.pt"
    # agnostic 1536
    # img_size = 1536
    img_size = 1280

elif detector == "yolov5":
    modelPathFront = '../Model/yolov5/yolov5x6.pt'
    img_sizeFront = 1280
    modelPath = '../Model/yolov5/yolov5x.pt'
    img_size = 640
    detectorType = None
    
else:
    print("detector not supported")
    exit(5)
