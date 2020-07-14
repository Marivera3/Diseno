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

video_getter.start()
time.sleep(2.0)
#for item in workers:
#    item.start()

# print('[INFO] Starting VideoGet...')
time.sleep(3.0)

## Set ffmeg instance
pathIn= './SavedImages/13/'
## REVISAR SI HAY UN FORMATO QUE SEA MAS COMPRIMIMDO
pathOut = 'video_v1.avi'
fps = 25
#SV = SaveVideo(name='VideoWriter', vg=video_getter, pathOut=pathIn+pathOut, fps=fps, encode_quality=95)

#print('[INFO] Starting saving Video...')
#SV.start()
ct = CentroidTracker(maxDisappeared=25, maxDistance=75)
trackers = []
trackableObjects = {}
out = 0
salidos = []
skipped_frames = 10

cpt=0;
fps_count = FPS().start()
while True:
	frame = video_getter.frame.copy()

	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	rects = []
	trackers = []

	if cpt % skipped_frames== 0:
		detections = get_faces(detector, embedder, frame, 0.5, fa)
		recon = []
		face_data = [(*face, *recognize(face[1], recognizer, le, 0.65)) for face in detections]
		for item in face_data:
			#Listas con nombres de reconocidos
			recon.append(item[3])
		for face in detections:
			(startX, startY, endX, endY) = face[2]
			tracker = dlib.correlation_tracker()
			rect = dlib.rectangle(startX, startY, endX, endY)
			tracker.start_track(rgb, rect)

			# add the tracker to our list of trackers so we can
			# utilize it during skip frames
			trackers.append(tracker)
			# loop over the trackers
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

	objects = ct.update(rects,recon)

	# loop over the tracked objects
	for (objectID, centroid_nombre) in objects.items():
		to = trackableObjects.get(objectID, None)
		if to is None:
			to = TrackableObject(objectID, centroid_nombre[0],centroid_nombre[1])
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)
			## Envio de mails
			if not to.sent:
				if to.reconocido:
					#enviar mail
					nombre = centroid_nombre[1]
				else:
					nombre = 'unknown '+string(objectID)
					# enviar mails
				to.sent = True

			# Si es que salio
			# check to see if the object has been counted or not
			if not to.counted and direction > 0  and centroid[1] > H - 50:
				out += 1
				to.counted = True
				salidos.append(centroid_nombre[1]+' salio')
		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

	for item in face_data:
		print(item[3])
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
	if cpt > 200:
		video_getter.stop()
		break
	exitbool = show_frame(frame)

#   if exitbool:
#       # SV.stop()
#       fps_count.stop()
#       print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
#       print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
#       time.sleep(1)
#       video_getter.stop()
#       # db_client.close()
#       break
