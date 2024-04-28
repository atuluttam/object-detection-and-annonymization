import os
import argparse
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument('-video',"--video",default = None, help="Enter the path to your video")
ap.add_argument('-output',"--output", help="Enter the IOU threshold for pedestrian")

args = vars(ap.parse_args())


cap = cv2.VideoCapture(args["video"])
width = int(cap.get(3))
height = int(cap.get(4))
frame_count = 0
start = time.time()


while(True):
    ret,frame = cap.read()      
    if ret==False:
        break
    cv2.imwrite(os.path.join(args["output"],"{}.png".format(frame_count)),frame)
    frame_count+=1

cap.release()


   
