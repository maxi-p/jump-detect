from Classifier import *
import os
import sys

def main():
    videoName = sys.argv[1]
    videoPath = os.path.join('test', videoName)
    configPath = os.path.join('model_data', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    modelPath = os.path.join('model_data', 'frozen_inference_graph.pb')
    classesPath = os.path.join('model_data', 'coco.names')

    classifier = Classifier(videoPath, configPath, modelPath, classesPath, videoName)
    classifier.onVideo()

if __name__ == '__main__':
    main()
