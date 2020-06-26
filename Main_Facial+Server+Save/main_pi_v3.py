###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 main_pi_v3.py --detector face_detection_model \
--embeddings output/embeddings.pickle \
--embedding-model openface_nn4.small2.v1.t7 \
--confidence 0.5 

\
--shape-predictor shape_predictor_68_face_landmarks.dat
'''

import json
import threading
import datetime as dt
import cv2
import argparse
import os
import hashlib
import io
import pickle
import codecs
import time
import sys
#import dlib
import numpy as np
import mongoengine as me
from pymongo import MongoClient
from subprocess import Popen, PIPE
from PIL import Image
from imutils.face_utils import FaceAligner
from functions_v4 import acquire_frame, draw_frame, show_frame, train
from VideoGet import VideoGet
from SaveVideo import SaveVideo
from User.User import PersonRasp



def addperson2db(name, surname, is_recongized, last_in, last_out, picture, likehood):
    idh = hashlib.sha256(str(time.time()).encode()).hexdigest()
    if last_out == "":

        #PersonRasp(idrasp=idh,name=name, surname=surname, last_out=last_out,is_recognized=is_recongized, likelihood=likehood).save()

        if is_recongized:
            PersonRasp(idrasp= idh,name=name, surname=surname, last_in=last_in,
                        is_recognized=is_recongized, likelihood=likehood).save()

        else:
            PersonRasp(idrasp= idh,name=name, surname=surname, last_in=last_in,
                        is_recognized=is_recongized, likelihood=likehood,
                        picture=picture).save()


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
#ap.add_argument("-p", "--shape-predictor", required=True,
#	            help="path to facial landmark predictor")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#predictor = dlib.shape_predictor(args["shape_predictor"])
#fa = FaceAligner(predictor, desiredFaceWidth=256)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

recognizer, le = train(data)

## INICIO ##

## Connect to DBs
# Connection to mongoengine DB Rasp
print('[INFO] Connecting to DB mongoengine')
#me.connect('person_rasp', host='mongodb://grupo14.duckdns.org:1226/Rasp')
#me.connect(host='mongodb://192.168.0.25:27017/Rasp', replicaset='rsdiseno')
me.connect('Rasp')
# Connect to pymongo
# mongo_client_Rasp = MongoClient('mongodb://grupo14.duckdns.org:1226')
# db_Rasp = mongo_client_Rasp["Test1"]
# col_Rasp = db_Rasp["person"]
# print(f"[INFO] Conecting to DB Rasp with:{col_Rasp.read_preference}...")

time.sleep(1.0)

# initialize the video stream, then allow the camera sensor to warm up

## Set Threding to start filming
video_getter = VideoGet(src=0, name='Video Getter')
time.sleep(1.0)
# print('[INFO] Starting VideoGet...')
video_getter.start()
time.sleep(3.0)

## Set ffmeg instance
pathIn= './SavedImages/13/'
## REVISAR SI HAY UN FORMATO QUE SEA MAS COMPRIMIMDO
pathOut = 'video_v1.avi'
fps = 25
#SV = SaveVideo(name='VideoWriter', vg=video_getter, pathOut=pathIn+pathOut, fps=fps, encode_quality=95)

#print('[INFO] Starting saving Video...')
#SV.start()

# cpt=0;
while True:
    # main_loop()

    frame = video_getter.frame.copy()

    face_data = acquire_frame(detector, embedder, frame , recognizer, le, 0.5, 0.65)#,fa)

    for item in face_data:
        if item[3] == 'unknown':
            pickled = codecs.encode(pickle.dumps(item[0]), "base64").decode()
            addperson2db(name='', surname='', is_recongized=False,
                        last_in=dt.datetime.utcnow(), last_out='',
                        picture=pickled, likehood=0)
        else:
            separador = item[3].index("_")
            addperson2db(name=item[3][0:separador],
                        surname=item[3][separador+1:-1], is_recongized=True,
                        last_in=dt.datetime.utcnow(), last_out='',
                        picture='', likehood=item[4])

        frame = draw_frame(frame, item)
    exitbool = show_frame(frame)




    if exitbool:
        # SV.stop()
        time.sleep(1)
        video_getter.stop()
        # db_client.close()
        break
