###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 main_pi_v3.py --detector face_detection_model \
--embeddings output/embeddings.pickle \
--embedding-model openface_nn4.small2.v1.t7 \
--confidence 0.5 \
--shape-predictor shape_predictor_68_face_landmarks.dat
'''

import json
import threading
import cv2
import argparse
import os
# import io
import pickle
import time
import sys
import dlib
import imutils
import numpy as np
import multiprocessing as mp
# from PIL import Image
from imutils.face_utils import FaceAligner
from imutils.video import FPS
from multiprocessing import Process, Queue, Manager
from pyimagesearch.centroidtracker_cp import CentroidTracker
from pyimagesearch.trackableobject_cp import TrackableObject
from functions_v4 import get_faces, recognize, draw_frame, show_frame, train
from VideoGetv3 import VideoGet
from SaveVideov2 import SaveVideo
# from FPS import FPS
from Person2DB import Person2DB
from CheckDB2 import CheckDB2
# from esp32_frame import esp32_frame

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
arg = vars(ap.parse_args())
# load our serialized face detector from disk


def main_core(args, frame_queue, pframe_queue):

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
    time.sleep(1.0)

    # initialize the video stream, then allow the camera sensor to warm up

    ## Set Threding to start filming
    video_getter = VideoGet(frame_queue=frame_queue, src=0, name='Video Getter')
    time.sleep(1.0)
    # print('[INFO] Starting VideoGet...')
    video_getter.start()
    time.sleep(1.0)


    ct = CentroidTracker(maxDisappeared=25, maxDistance=75)
    trackers = []
    trackableObjects = {}
    out = 0
    skipped_frames = 2
    out_prev = 0

    cpt=0;
    exitbool = False
    fps_count = FPS().start()
    while True:
        frame = video_getter.frame.copy()
    	# frame = esp32_frame("grupo14.duckdns.org", 1228)
    	# if frame is None:
    	# 	continue
    	# frame = np.array(frame)
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
        	detections = get_faces(detector, embedder, frame, 0.5, fa)
        	#[(face,vector,coordenada,imagen_completa)]
        	face_data = [(*face, *recognize(face[1], recognizer, le, 0.65)) for face in detections]
        	#[(face,vector,coordenada,imagen_completa, nombre, prob)]
        	for item in face_data:
        		#Listas con nombres de reconocidos
        		recon.append(item[4])
        		fotos.append(item[0])
        		ps.append(item[5])
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

        objects, names, images, probabilities  = ct.update(rects,recon, fotos, ps)
        # loop over the tracked objects
        for (objectID, centroid),(ID, name),(I,im),(D,prob) in zip(objects.items(),
        					names.items(), images.items(), probabilities.items()):

        	to = trackableObjects.get(objectID, None)
        	if to is None:
        		to = TrackableObject(objectID, centroid, name, im, prob)
        	else:
        		y = [c[1] for c in to.centroids]
        		direction = centroid[1] - np.mean(y)
        		to.centroids.append(centroid)
        		# Si es que salio
        		# check to see if the object has been counted or not
        		if not to.counted and direction > 0  and centroid[1] > H - 50:
        			to.out = True
        			to.counted = True
        	# store the trackable object in our dictionary
        	trackableObjects[objectID] = to

        	#Coordinamos el paquete de envio
        	## Envio de mails
        	if not to.sent:
        		paquete = [to.prob, to.pic, to.reconocido, to.out]
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

        #   frame = draw_frame(frame, item)
        fps_count.update()
        cpt += 1
        out_prev = out
        # if cpt > 250:
        # 	video_getter.stop()
        # 	break
        exitbool = 0# show_frame(frame)
        if exitbool or cpt > 50:
          # SV.stop()
          fps_count.stop()
          print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
          print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
          time.sleep(1)
          video_getter.stop()
          # db_client.close()
          break


def sec_core(frame_queue):


    # Set ffmeg instance
    print('[INFO] Starting:', mp.current_process().name)
    pathIn= 'BackupVideo/13/'
    # ## REVISAR SI HAY UN FORMATO QUE SEA MAS0.2497033993641794 COMPRIMIMDO
    pathOut = 'video_v1.avi'
    fps = 30
    SV = SaveVideo(name='VideoWriter', vg=frame_queue, pathOut=pathIn+pathOut, fps=fps, encode_quality=95)

    print('[INFO] Starting saving Video...')
    SV.start()

def third_core(queue_pframes):
    print('[INFO] Starting:', mp.current_process().name)
    DB = Person2DB(name='Person2db', queue_pframes=queue_pframes)
    time.sleep(1)
    DB.start()

    print('[INFO] Starting to check server DB')

    serverdb = CheckDB2(name='CHeckDB2', seconds=2)
    serverdb.start()

    while True:
        if serverdb.has_changes:
            print('Hay Database para actualizar')
        time.sleep(1)








if __name__ == "__main__":

    # main_core(detector, embedder, recognizer, le)

    frame_queue = mp.Queue()
    pframe_queue = mp.Queue()

    p = mp.Process(target=main_core, args=(arg, frame_queue, pframe_queue,))
    sec = mp.Process(target=sec_core, args=(frame_queue,))
    # third = mp.Process(target=third_core, args=(pframe_queue,))
    p.start()
    time.sleep(1)
    sec.start()
    # sec.join()
    # time.sleep(5)
    # third.start()
    # third.join()
    # p.join()

    # [pframe_queue.get() for _ in range(pframe_queue.qsize())]
