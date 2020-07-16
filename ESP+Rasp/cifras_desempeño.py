###############################################################################
# SCRIPT PRINCIPAL DE RECONOCIMIENTO FACIAL                                   #
# Diseño Eléctrico - Grupo 14                                                 #
###############################################################################

'''
python3 cifras_desempeño.py --dataset dataset --detector face_detection_model \
--embeddings output/embeddings_v2.pickle \
--embedding-model dlib_face_recognition_resnet_model_v1.dat \
--confidence_dec 0.5 --confidence_rec 0.5 \
--shape-predictor shape_predictor_68_face_landmarks.dat \
--shape-pred shape_predictor_5_face_landmarks.dat

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
import dlib
import numpy as np
import imutils
from PIL import Image
from imutils import paths
from imutils.face_utils import FaceAligner
from functions_v4 import get_faces, recognize, draw_frame, show_frame, train
from VideoGet import VideoGet
from FrameProcessing import FrameProcessing
from imutils.video import FPS

# PARAMETERS

# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-d", "--detector", required=True,
				help="path to OpenCV's deep learning face detector")
ap.add_argument("-e", "--embeddings", required=True,
				help="path to serialized db of facial embeddings")
ap.add_argument("-m", "--embedding-model", required=True,
				help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c_d", "--confidence_dec", type=float, default=0.5,
				help="minimum probability to filter weak detections")
ap.add_argument("-c_r", "--confidence_rec", type=float, default=0.5,
				help="minimum probability to filter weak identificactions")
ap.add_argument("-p", "--shape-predictor", required=True,
				help="path to facial landmark predictor")
ap.add_argument("-b", "--shape-pred", required=True,
				help="path to facial land for vector")

args = vars(ap.parse_args())


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
							  "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
confianza_dec = args["confidence_dec"]
confianza_recon = args["confidence_rec"]

predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = dlib.face_recognition_model_v1(args["embedding_model"])
sp = dlib.shape_predictor(args["shape_pred"])
# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# grab the paths to the input images in our dataset
i = 0
for (rootDir, dirNames, filenames) in os.walk(args["dataset"]):
	if i == 0:
		imagePaths = dirNames
	i += 1

recognizer, le = train(data)


video_getter = VideoGet(src = 0, name='Video Getter')

#workers = [FrameProcessing(i, q_video, q_frame, detector, embedder, recognizer, le, fa) for i in range(1, 3)]
print('[INFO] Starting VideoGet...')
video_getter.start()
time.sleep(2.0)

frames_per_person = 100
calculos = {}
fps_count = FPS().start()

for name in imagePaths:
	Coincidencias = 0
	Caras = 0
	print('Persona a evaluar : {}'.format(name))
	print('Prepare imagen')
	input('Presione cualquier boton')
	for j in range(4):
		for i in range(frames_per_person):
			frame = video_getter.frame.copy()
			#frame = imutils.resize(frame, width=500)
			detections = get_faces(detector, embedder, sp,frame, confianza_dec, fa)
			#[(face,vector,coordenada,imagen_completa)]
			face_data = [(*face, *recognize(face[1], recognizer, le, confianza_recon)) for face in detections]
			#[(face,vector,coordenada,imagen_completa, nombre, prob)]
			for item in face_data:
				print(item[4])
				if item[4] == name:
					Coincidencias += 1
				Caras += 1
			show_frame(frame)
			fps_count.update()
		input('Cambio de foto')
	calculos[name] = (Coincidencias,Caras)

print(calculos)


fps_count.stop()
print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
time.sleep(1)
video_getter.stop()
