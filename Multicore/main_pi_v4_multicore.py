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
#import dlib
import numpy as np
import multiprocessing as mp
# from PIL import Image
from imutils.face_utils import FaceAligner
from imutils.video import FPS
from functions_v4 import acquire_frame, draw_frame, show_frame, train
from VideoGetv2 import VideoGet
from SaveVideov2 import SaveVideo
# from FPS import FPS
from Person2DB import Person2DB
from CheckDB2 import CheckDB2


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
arg = vars(ap.parse_args())

# load our serialized face detector from disk


def main_core(args, frame_queue, pframe_queue):

    print('[INFO] Starting:', mp.current_process().name)

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
    time.sleep(1.0)

    # initialize the video stream, then allow the camera sensor to warm up

    ## Set Threding to start filming
    video_getter = VideoGet(frame_queue=frame_queue, src=0, name='Video Getter')
    time.sleep(1.0)
    # print('[INFO] Starting VideoGet...')
    video_getter.start()
    time.sleep(1.0)

    cpt=0;
    exitbool = False
    fps_count = FPS().start()

    while True:

        frame = video_getter.frame.copy()


        face_data = acquire_frame(detector, embedder, frame , recognizer, le, 0.5, 0.65)#,fa)
        # pframe_queue.put(face_data)
        for item in face_data:
            # print(item[2:])
            frame = draw_frame(frame, item)
        fps_count.update()

        cpt +=1

        exitbool = show_frame(frame)

        if exitbool or cpt == 80:
            #
            fps_count.stop()
            print("[INFO] elasped time fps processed: {:.2f}".format(fps_count.elapsed()))
            print("[INFO] approx. processed FPS: {:.2f}".format(fps_count.fps()))
            time.sleep(1)
            video_getter.stop()
            time.sleep(2)
            print('[INFO] Exiting :', mp.current_process().name)
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
    # sec = mp.Process(target=sec_core, args=(frame_queue,))
    # third = mp.Process(target=third_core, args=(pframe_queue,))
    p.start()
    time.sleep(10)
    # sec.start()
    # sec.join()
    # time.sleep(5)
    # third.start()
    # third.join()
    # p.join()

    # [pframe_queue.get() for _ in range(pframe_queue.qsize())]
