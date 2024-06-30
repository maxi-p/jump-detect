import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

class Classifier:
    def __init__(self,videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        self.readClasses()

        self.air = load_model(os.path.join('models', 'imageclassifier.h5'))
        self.state = ['air', 'land']

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened() == False):
            print("error")
            return 
            
        (success, image) = cap.read()

        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.3)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold= 0.5, nms_threshold= 0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    if (classLabelIDs[i] == 1) :
                        bbox = bboxs[np.squeeze(bboxIdx[i])]
                        classConfidence = confidences[np.squeeze(bboxIdx[i])]
                        classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                        classLabel = self.classesList[classLabelID]
                        x,y,w,h = bbox

                        x = int(x - 0.1*w)
                        y = int(y - 0.1*h)
                        w = int(1.2*w)
                        h = int(1.2*h)

                        y_start = max(0, y)
                        y_end = min(image.shape[0], y+h)
                        x_start = max(0, x)
                        x_end = min(image.shape[1], x+w)

                        cropped_image = image[y_start:y_end, x_start:x_end]
                        resize = tf.image.resize(cropped_image, (256,256))
                        yhat = self.air.predict(np.expand_dims(resize/255, 0))
                        if yhat > 0.3:
                            cv2.putText(image, 'land', (10,450), self.font, 3, (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(image, 'AIR', (10,450), self.font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(image, (x_start,y_start), (x_end, y_end), color=(255,255,255), thickness=1)

            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()     
