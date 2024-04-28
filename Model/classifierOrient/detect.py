from keras.preprocessing import image
import numpy as np
import os
import argparse
import pandas as pd
from tensorflow.keras.models import Sequential, save_model, load_model
    
def load_model_keras(path_to_model):
    loaded_model = load_model(
        path_to_model,
        custom_objects=None,
        compile=False
    )
    return loaded_model

def preprocess_input(x, dim_ordering='default'):
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    return x


def single_image_detect(model,path,classes):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    prediction = np.argmax(predictions)
    csv = [[path , classes[prediction]]]
    return csv


def batch_detect(path,model,classes,batch_size=8,target_size=(224,224)):
    total_images = os.listdir(path)
    csv = []
    print("Processing {} images found in directory".format(len(total_images)))
    batches = int(len(total_images)/batch_size) + 1
    for batch in range(batches):
        images = []
        image_names = []
        for ind,img in enumerate(total_images[batch*batch_size:(batch+1)*batch_size]):
            if not(os.path.isdir(img)):
                image_x = image.load_img(os.path.join(path,img), target_size=(224, 224))
                image_x = image.img_to_array(image_x)
                image_x = preprocess_input(image_x)
                images.append(image_x)
                image_names.append(img)
            else:
                ind-=1
        images = np.array(images)
        if len(image_names) == 1:
            predictions = model.predict(images)
            prediction = np.argmax(predictions)
            csv.append([image_names[0],classes[prediction]])
        elif len(image_names) > 1:
            predictions = model.predict(images)
            assert len(predictions) == len(image_names), print(image_names,predictions.shape)
            prediction = np.argmax(predictions,axis=1)
            
            for pred in range(len(prediction)):
                csv.append([image_names[pred],classes[prediction[pred]]])
    return csv




ap = argparse.ArgumentParser()
ap.add_argument('-path',"--path",required=True, help="Enter the path to your images")
ap.add_argument('-dest',"--dest",default="output", help="Enter the path to save your csv")
ap.add_argument('-model',"--model",required=True, help="Enter the path to the trained model")
args = vars(ap.parse_args())
classes = ['Front','Side','Back']

if not os.path.exists(args["dest"]):
    os.mkdir(args["dest"])

model = load_model_keras(args["model"])
print("model loaded successfully")
import time
st = time.time()
if os.path.isdir(args["path"]):
    csv = batch_detect(args["path"],model,classes=classes)

else:
    csv = single_image_detect(model,args["path"],classes=classes)
print("time",time.time()-st)
df = pd.DataFrame(csv,columns=["file_name" , "detected_class"])
df.to_csv(os.path.join(args["dest"],"predictions.csv"),index=False)
print("Output file : predicted.csv saved in folder : {}".format(args["dest"]))
            
                                
            
                                
        
    

    


