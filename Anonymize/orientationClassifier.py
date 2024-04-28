from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from utils import hyper_parameters as hp
import cv2
import tensorflow as tf


class orientationClassifier:
    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        self.classes = ["Front","Side","Back"]
        self.model = load_model(
        hp.classifierModelWeights,
        custom_objects=None,
        compile=False
    )

    def preprocess_input(self,x):
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        return x

    def detect(self,image):
        image = cv2.resize(image,(224,224))
        x = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        x = x.astype('float32')
        x = self.preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        predictions = self.model.predict(x)
        prediction = np.argmax(predictions)
        return self.classes[prediction],np.max(predictions,axis=-1)[0]

    def batch_detect(self,images,batch_size=8,target_size=(224,224)):
        classOutputs = []
        imagesProcess = []
        for image in images:
            image = cv2.resize(image,(224,224))
            x = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            x = x.astype('float32')
            x = self.preprocess_input(x)
            imagesProcess.append(x)
        imagesProcess = np.array(imagesProcess)
        predictions = self.model.predict(imagesProcess)
        prediction = np.argmax(predictions,axis=1)
        predictionProb = np.max(predictions,axis=1)
        for pred in range(len(prediction)):
            classOutputs.append(self.classes[prediction[pred]])
        return np.array(classOutputs),predictionProb