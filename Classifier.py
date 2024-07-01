import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

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

        self.mp_holistic = mp.solutions.holistic # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities

        self.font = cv2.FONT_HERSHEY_SIMPLEX

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

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while success:
                if endFrame != 0:
                    if recordingFrame == endFrame:
                        endFrame = 0
                        # write the clip
                    else:
                        # add frame to out
                        outClip.write(recordingImage)

                ###################
                if gx1 != 0 and gy1 !=0 and state == RELEASED:
                    image, results = self.mediapipe_detection(image, holistic)
            
                    self.draw_styled_landmarks(image, results)

                    if results.pose_landmarks and self.inBox(results.pose_landmarks):
                        if endFrame == 0:
                            endFrame = frame + 30
                            outClip = cv2.VideoWriter('./clips/'+self.videoName+'/CLIP-'+str(writeCounter)+'.mp4', self.codec, 30, (self.width, self.height))
                            writeCounter += 1
                        cv2.putText(image, 'AIR', (10,450), self.font, 3, (0, 255, 0), 2, cv2.LINE_AA)

                ###################

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
    
    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image.flags.writeable = False                  
        results = model.process(image)                
        image.flags.writeable = True                    
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
    
    def inBox(self, pose_landmarks):
        arms = []
        for i in range(12,16):
            arms.append([pose_landmarks.landmark[i].x*self.width, pose_landmarks.landmark[i].y*self.height])

        for point in arms:
            if point[0] >= gx1 and point[0] <= gx2 and point[1] >= gy1 and point[1] <= gy2:
                return True

        return False

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
            tempX = max(gx1, gx2)
            tempY = max(gy1, gy2)
            gx1 = min(gx1, gx2)
            gy1 = min(gy1, gy2)
            gx2 = tempX
            gy2 = tempY
            print('(',gx1,',',gy1,'), (',gx2,',',gy2,')')
            state = RELEASED
