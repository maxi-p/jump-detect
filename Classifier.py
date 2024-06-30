import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

RELEASED, DESELECT = 1,1
gx1, gx2, gy1, gy2, gxTemp, gyTemp, PRESSED, SELECT = 0,0,0,0,0,0,0,0
state = RELEASED
selection_state = DESELECT

class Classifier:
    def __init__(self,videoPath, configPath, modelPath, classesPath, videoName):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.height = 0
        self.width = 0
        self.videoName = ''
        self.codec = cv2.VideoWriter_fourcc(*'MP4V')

        for char in videoName:
            if char != '.':
                self.videoName += char
            else:
                break


        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        self.readClasses()

        # self.air = load_model(os.path.join('models', 'imageclassifier.h5'))
        self.state = ['air', 'land']

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')  

    def createDir(self, dirname):
        dirs = os.listdir('./clips/')
        for dir in dirs:
            if dir == dirname:
                return
        os.mkdir('./clips/'+dirname)

    def onVideo(self):
        global selection_state
        
        self.createDir(self.videoName)
        
        cap = cv2.VideoCapture(self.videoPath)
        recordingCap = cv2.VideoCapture(self.videoPath)
        
        if cap.isOpened() == False or recordingCap.isOpened() == False:
            print("error")
            return 
            
        (success, image) = cap.read()
        self.height = image.shape[0]
        self.width = image.shape[1]
        recordingSuccess, recordingImage, outClip = None, None, None
        endFrame = 0
        writeCounter = 1
        frame = 0
        recordingFrame = -29

        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.3)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold= 0.5, nms_threshold= 0.2)

            if endFrame != 0:
                if recordingFrame == endFrame:
                    endFrame = 0
                    # write the clip
                else:
                    # add frame to out
                    outClip.write(recordingImage)
        
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

                        if inBox(x_start, y_start, x_end, y_end, gx1, gy1, gx2, gy2):
                            if endFrame == 0:
                                endFrame = frame + 30
                                outClip = cv2.VideoWriter('./clips/'+self.videoName+'/CLIP-'+str(writeCounter)+'.mp4', self.codec, 30, (self.width, self.height))
                                writeCounter += 1
                            cv2.putText(image, 'AIR', (10,450), self.font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(image, (x_start,y_start), (x_end, y_end), color=(255,255,255), thickness=2)

            cv2.rectangle(image, (gx1, gy1), (gx2, gy2), color=(255,255,0), thickness=2)
            cv2.imshow('scan', image)
            cv2.setMouseCallback('scan', mouseCallback)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1)
            if key == ord('s'):
                selection_state = SELECT
            if key == ord('d'):
                selection_state = DESELECT

            (success, image) = cap.read()
            if recordingFrame >= 0:
                (recordingSuccess, recordingImage) = recordingCap.read()
            
            frame += 1
            recordingFrame += 1
        cv2.destroyAllWindows()   

def mouseCallback(event, x, y, flags, params):
    global gx1, gy1, gx2, gy2, state, selection_state
    if selection_state == SELECT:
        if state == PRESSED:
            gx2 = x
            gy2 = y
        if event == cv2.EVENT_LBUTTONDOWN:
            gx1 = x
            gy1 = y
            gx2 = x
            gy2 = y
            state = PRESSED
        if event == cv2.EVENT_LBUTTONUP:
            state = RELEASED

def inBox(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    w1 = abs(ax2 - ax1)
    w2 = abs(bx2 - bx1)
    h1 = abs(ay2 - ay1)
    h2 = abs(by2 - by1)
    
    if ax1 < ax2:
        mx1 = ax1+int(w1/2)
    else:
        mx1 = ax2+int(w1/2)
    if ay1 < ay2:
        my1 = ay1+int(h1/2)
    else:
        my1 = ay2+int(h1/2)
    if bx1 < bx2:
        mx2 = bx1+int(w2/2)
    else:
        mx2 = bx2+int(w2/2)
    if by1 < by2:        
        my2 = by1+int(h2/2)
    else:
        my2 = by2+int(h2/2)

    dx = abs(mx1 - mx2)
    dy = abs(my1 - my2)

    if (dx < int((w1 + w2)/2)) and (dy < int((h1 + h2)/2)):
        return True
    
    return False

