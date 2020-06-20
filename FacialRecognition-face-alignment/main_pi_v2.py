###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

import json
import threading
import datetime
import cv2
import argparse
import os
import pickle
import time
from imutils.video import VideoStream
from imutils.video import FPS
from functions_v4 import acquire_frame, draw_frame, show_frame, train
from VideoGet import VideoGet
from VideoShow import VideoShow
# PARAMETERS

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-e", "--embeddings", required=True,
                help="path to serialized db of facial embeddings")
ap.add_argument("-m", "--embedding-model", required=True,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

recognizer, le = train(data)

# THREADING

class CameraLoop(threading.Thread):
    '''
    Comenzar loop de reconocimiento en la cámara CAM
    '''
    def __init__(self, threadID, CAM=None, name=None):
        threading.Thread.__init__(self)
        self.threadid = threadID
        self.cam = CAM
        self.name = name
    def run(self):
        print("Starting camera loop on {self.name}")
        while True:
            main_loop()

class DatabaseLoop(threading.Thread):
    '''
    Comenzar loop de base de datos
    '''
    def __init__(self, threadID, name):
        super().__init__(self)
        self.threadid = threadID
        self.name = name

def db_append():
    '''
    Añade el vector y la foto a la base de datos
    '''
    pass

def db_retrieve():
    '''
    Consulta la base de datos
    '''
    pass

def main_loop():

    frame, face_data = acquire_frame(detector, embedder, vs, recognizer, le,
            0.5, 0.65)
    for item in face_data:
        frame = draw_frame(frame, item)
    show_frame(frame)


## INICIO ##

# initialize the video stream, then allow the camera sensor to warm up
# print("[INFO] starting viif key == ord('q') or key == 27: self.stopped = Truedeo stream...")
# vs = VideoStream(src=0).start()
video_getter = VideoGet(src=0, name='Video Getter')
time.sleep(1.0)
video_getter.start()
time.sleep(1.0)

while True:
    # main_loop()

    frame = video_getter.frame
    face_data = acquire_frame(detector, embedder, frame , recognizer, le,
            0.5, 0.65)

    for item in face_data:
        frame = draw_frame(frame, item)
    exitbool = show_frame(frame)



    # exitbool = show_frame(frame);
    if exitbool:
        video_getter.stop()
        break
