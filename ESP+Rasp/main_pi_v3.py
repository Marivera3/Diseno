###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 main_pi_v3.py --detector face_detection_model \
--embeddings output/embeddings.pickle \
--embedding-model openface_nn4.small2.v1.t7 \
--confidence 0.5 --shape-predictor shape_predictor_68_face_landmarks.dat
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
import datetime
import sys
from multiprocessing import Process, Queue, Manager
from pyimagesearch.centroidtracker_cp import CentroidTracker
from pyimagesearch.trackableobject_cp import TrackableObject
import dlib
import numpy as np
import imutils
import mongoengine as me
from pymongo import MongoClient
from subprocess import Popen, PIPE
from PIL import Image
from imutils.face_utils import FaceAligner
from functions_v4 import get_faces, recognize, draw_frame, show_frame, train
from VideoGet import VideoGet
from SaveVideo import SaveVideo
from User.User import PersonRasp
from FrameProcessing import FrameProcessing
from imutils.video import FPS
from esp32_frame import esp32_frame



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
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)
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
pathIn= './videorecordings/13/'
# pathIn = './videorecordings/' + str(datetime.datetime.now().hour) + '/'
## REVISAR SI HAY UN FORMATO QUE SEA MAS COMPRIMIMDO
pathOut = 'video_v1.avi'
fps = 25
SV = SaveVideo(name='VideoWriter', vg=video_getter, pathOut=pathIn+pathOut, fps=fps, encode_quality=95)

print('[INFO] Starting saving Video...')
SV.start()
ct = CentroidTracker(maxDisappeared=25, maxDistance=75)
trackers = []
trackers_esp32 = []
trackableObjects = {}
out = 0
skipped_frames = 2
out_prev = 0

cpt=0;
fps_count = FPS().start()
while True:
    frame = video_getter.frame.copy()

    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (H, W) = frame.shape[:2]
    rects = []

    #print(cpt)
    if cpt % skipped_frames== 0:
        recon = []
        fotos = []
        ps = []
        trackers = []
        trackers_esp32 = []
        devices = []

        detections = get_faces(detector, embedder, frame, 0.5, fa)
        #[(face,vector,coordenada,imagen_completa)]
        face_data = [(*face, *recognize(face[1], recognizer, le, 0.65)) for face in detections]
        #[(face,vector,coordenada,imagen_completa, nombre, prob)]
        for item in face_data:
            #Listas con nombres de reconocidos
            recon.append(item[4])
            fotos.append(item[0])
            ps.append(item[5])
            [devices.append(1) for i in range(len(face_data))]
        for face in detections:
            (startX, startY, endX, endY) = face[2]
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
            # loop over the trackers
        '''

        frame_esp32 = esp32_frame('grupo14.duckdns.org', 1228)
        if frame_esp32:
            rgb_esp32 = cv2.cvtColor(frame_esp32, cv2.COLOR_BGR2RGB)
            detections_esp32 = get_faces(detector, embedder, frame_esp32, 0.5, fa)
            face_data_esp32 =  [(*face, *recognize(face[1], recognizer, le, 0.65)) for face in detections_esp32]
            for item in face_data_esp32:
                    recon.append(item[4])
                    fotos.append(item[0])
                    ps.append(item[5])
                    [devices.append(0) for i in range(len(face_data_esp32))]
            for face in detections_esp32:
                    (startX, startY, endX, endY) = face[2]
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb_esp32, rect)
                    trackers_esp32.append(tracker)
                    # loop over the trackers
        '''
    else:
        for tracker in trackers:
            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
            # update the tracker and grab the updated position
            '''
            frame_esp32 = esp32_frame('grupo14.duckdns.org', 1228)
            if frame_esp32:
                rgb_esp32 = cv2.cvtColor(frame_esp32, cv2.COLOR_BGR2RGB)
                for tracker in trackers_esp32:
                    tracker.update(rgb_esp32)
                    pos = tracker.get_position()
                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))
            '''

    objects, names, images, probabilities, devicess  = ct.update(rects, recon, fotos, ps, devices)
    # loop over the tracked objects
    for (objectID, centroid),(ID, name),(I,im),(D,prob), (F,dev) in zip(objects.items(),
        names.items(), images.items(), probabilities.items(), devicess.items()):

        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid, name, im, prob, dev)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # Si es que salio
            # check to see if the object has been counted or not
            if not to.counted and direction > 0  and centroid[1] > H - 50 and to.device == 1:
                to.out = True
                to.counted = True
            elif not to.counted and direction > 0  and centroid[1] > H - 250 and to.device == 0:
                to.inn = True
                to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        #Coordinamos el paquete de envio
        ## Envio de mails
        if not to.sent:
            paquete = [to.prob, to.pic, to.reconocido, to.out, to.inn]
            if to.reconocido:
                #enviar mail
                paquete.append(to.name)
            else:
                paquete.append('unknown {}'.format(objectID))
                # enviar mails
            ##Paquete a enviar
            to.sent = True

    for item in face_data:
        print('Reconocido ',item[4])
#       if item[3] == 'unknown':
#           pickled = codecs.encode(pickle.dumps(item[0]), "base64").decode()
#           addperson2db(name='', surname='', is_recongized=False,
#                       last_in=dt.datetime.utcnow(), last_out='',
#                       picture=pickled, likehood=0)
#       else:
#           separador = item[3].index("_")
#           addperson2db(name=item[3][0:separador],
#                       surname=item[3][separador+1:-1], is_recongized=True,
#                       last_in=dt.datetime.utcnow(), last_out='',
#                       picture='', likehood=item[4])

#   frame = draw_frame(frame, item)
    fps_count.update()
    cpt += 1
    out_prev = out
    # if cpt > 250:
    #     video_getter.stop()
    #     break
    exitbool = 0#show_frame(frame)
    if exitbool or cpt > 100:
         SV.stop()
         fps_count.stop()
         print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
         print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
         time.sleep(1)
         video_getter.stop()
         # db_client.close()
         break
