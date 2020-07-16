###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 main_pi_v5.py --detector face_detection_model \
--embeddings output/embeddings.pickle \
--embedding-model openface_nn4.small2.v1.t7 \
--confidence 0.5 --shape-predictor shape_predictor_68_face_landmarks.dat
'''

import json
import threading
import datetime
import cv2
import argparse
import os
import pickle

import time
import sys
import numpy as np
import mongoengine as me
from subprocess import Popen, PIPE
from PIL import Image
from imutils.face_utils import FaceAligner
from functions_v4 import acquire_frame, draw_frame, show_frame, train
from VideoGetv2 import VideoGet
from SaveVideov3 import SaveVideo
from imutils.video import FPS



# PARAMETERS

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--detector", required=True,
# 				help="path to OpenCV's deep learning face detector")
# ap.add_argument("-e", "--embeddings", required=True,
# 				help="path to serialized db of facial embeddings")
# ap.add_argument("-m", "--embedding-model", required=True,
# 				help="path to OpenCV's deep learning face embedding model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 				help="minimum probability to filter weak detections")
# ap.add_argument("-p", "--shape-predictor", required=True,
# 				help="path to facial landmark predictor")
# args = vars(ap.parse_args())
#
# # load our serialized face detector from disk
# print("[INFO] loading face detector...")
# protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
# modelPath = os.path.sep.join([args["detector"],
# 							  "res10_300x300_ssd_iter_140000.caffemodel"])
# detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#
# predictor = dlib.shape_predictor(args["shape_predictor"])
# fa = FaceAligner(predictor, desiredFaceWidth=256)
# # load our serialized face embedding model from disk
# print("[INFO] loading face recognizer...")
# embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
#
# # load the face embeddings
# print("[INFO] loading face embeddings...")
# data = pickle.loads(open(args["embeddings"], "rb").read())
#
# recognizer, le = train(data)

## INICIO ##

## Connect to DBs
# Connection to mongoengine DB Rasp
# print('[INFO] Connecting to DB mongoengine')
#me.connect('person_rasp', host='mongodb://grupo14.duckdns.org:1226/Rasp')
#me.connect(host='mongodb://192.168.0.25:27017/Rasp', replicaset='rsdiseno')
#me.connect('Rasp')
# Connect to pymongo
# mongo_client_Rasp = MongoClient('mongodb://grupo14.duckdns.org:1226')
# db_Rasp = mongo_client_Rasp["Test1"]
# col_Rasp = db_Rasp["person"]
# print(f"[INFO] Conecting to DB Rasp with:{col_Rasp.read_preference}...")

time.sleep(1.0)

# initialize the video stream, then allow the camera sensor to warm up

## Start processes

video_getter = VideoGet(src = 0, name='Video Getter')

#workers = [FrameProcessing(i, q_video, q_frame, detector, embedder, recognizer, le, fa) for i in range(1, 3)]
print('[INFO] Starting VideoGet...')
video_getter.start()
time.sleep(2.0)
#for item in workers:
#    item.start()


## Set ffmeg instance
pathIn= './videorecordings/'
# pathIn = './videorecordings/' + str(datetime.datetime.now().hour) + '/'
## REVISAR SI HAY UN FORMATO QUE SEA MAS COMPRIMIMDO
pathOut = '/video_v1.avi'
fps = 25
SV = SaveVideo(name='VideoWriter', vg=video_getter, pathIn=pathIn, pathOut=pathOut, fps=fps, encode_quality=95)

print('[INFO] Starting saving Video...')
SV.start()

cpt=0;
fps_count = FPS().start()
while True:
	frame = video_getter.frame.copy()
	time.sleep(1)

	fps_count.update()
	exitbool = show_frame(frame)
	# cpt += 1

	if exitbool or cpt > 100:
		 SV.stop()
		 fps_count.stop()
		 print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
		 print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
		 time.sleep(1)
		 video_getter.stop()
		 # db_client.close()
		 break
