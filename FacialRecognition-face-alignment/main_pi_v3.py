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
import sys
import numpy as np
import mongoengine as me
from subprocess import Popen, PIPE
from PIL import Image
from functions_v4 import acquire_frame, draw_frame, show_frame, train
from VideoGet import VideoGet
from SaveVideo import SaveVideo


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



## K meas 4 image reduction

def initialize_K_centroids(X, K):
    """ Choose K points from X at random """
    m = len(X)
    return X[np.random.choice(m, K, replace=False), :]

def find_closest_centroids(X, centroids):
    m = len(X)
    c = np.zeros(m)
    for i in range(m):
        # Find distances
        distances = np.linalg.norm(X[i] - centroids, axis=1)

        # Assign closest cluster to c[i]
        c[i] = np.argmin(distances)

    return c

def compute_means(X, idx, K):
    _, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        examples = X[np.where(idx == k)]
        mean = [np.mean(column) for column in examples.T]
        centroids[k] = mean
    return centroids

def find_k_means(X, K, max_iters=10):
    centroids = initialize_K_centroids(X, K)
    previous_centroids = centroids
    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_means(X, idx, K)
        if (centroids == previous_centroids).all():
            # The centroids aren't moving anymore.
            return centroids
        else:
            previous_centroids = centroids

    return centroids, idx

def applyKKmean(frame, K=20, maxiters=20):
    X = frame.reshape((640*480, 3))
    colors, _ = find_k_means(X, K, max_iters=maxiters)
    idx = find_closest_centroids(X, colors)
    idx = np.array(idx, dtype=np.uint8)
    X_reconstructed = np.array(colors[idx, :] * 255, dtype=np.uint8).reshape((640,480, 3))
    compressed_image = Image.fromarray(X_reconstructed)
    return compressed_image

## INICIO ##

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
pathOut = 'video_pip_v13.avi'
fps = 25
# SV = SaveVideo(name='VideoWriter', vg=video_getter, pathOut=pathIn+pathOut, fps=fps, encode_quality=95)
# time.sleep(1.0)
# ## Starting saving video
# decode:jpeg2000,
# encode: mpeg4
process = Popen(['ffmpeg','-y', '-f', 'image2pipe','-vcodec', 'mjpeg',
                '-use_wallclock_as_timestamps', '1', '-loglevel', 'error',
                '-i', '-', '-vcodec', 'mpeg4', '-pix_fmt', 'yuv420p',
                '-r', str(fps), pathIn+pathOut], stdin=PIPE)
print('[INFO] Starting saving Video...')
# SV.start()
# encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
# cpt=0;
while True:
    # main_loop()

    frame = video_getter.frame.copy()

    ## Caso 1
    data = cv2.imencode('.jpg', frame, encode_param)[1].tostring()
    process.stdin.write(data)
    # print(f'data:{data}')
    # print(f'Size: {np.shape(data)}')
    # print(f'Max: {np.max(data)}')
    # print(f'Min: {np.min(data)}')
    # FramesVideo(frame=Binary(data, subtype=128 )).save()

    ## Caso 2

    # cv2.imwrite(os.path.join('SavedImages/17', f'Hour:{datetime.datetime.utcnow()}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

    ## Caso 3




    # face_data = acquire_frame(detector, embedder, frame , recognizer, le,
    #         0.5, 0.65)
    #
    # for item in face_data:
    #     frame = draw_frame(frame, item)
    exitbool = show_frame(frame)
    #frame2save = applyKKmean(frame, K=20, maxiters=20);


    if exitbool:
        # SV.stop()
        process.stdin.close()
        time.sleep(1)
        video_getter.stop()
        # db_client.close()
        break
