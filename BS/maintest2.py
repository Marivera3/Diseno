# https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np
import threading
from showmotion import showmotion
from VideoGet import VideoGet
from VideoShow import VideoShow
import time
from CountsPerSec import CountsPerSec

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

face_cascade = cv.CascadeClassifier(
    'haarcascade/haarcascade_frontalface_default.xml')

detectmov = True
square = False


# select the model of BS
# Parameters: history, varThreshold, bShadowDetection
backSub = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=50)
# backSub = cv.createBackgroundSubtractorKNN()
video_getter = VideoGet(0).start()
time.sleep(2)
video_shower_camara = VideoShow(video_getter.frame, "Camera").start()

#video_shower_MotionDetection = VideoShow(backSub.apply(
    #video_getter.frame), "Motion Detection").start()

while True:

    frame = video_getter.frame
    fgmask = backSub.apply(frame)

    if detectmov:
        # mov: Treu if movement is detected; nomd: number of motion detected
        _, mov = showmotion(fgmask.copy(), frame.copy(), square=square)

        if mov:
            # print('processing faces...')
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.3, 5)  # depend on the size of the image
            # faces is a list of list that each sub list give the pixel position of the edges of the frame
            [cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             for (x, y, w, h) in faces]
        else:
            print('no movements')

    # Update frames
    #video_shower_MotionDetection.frame = fgmask

    if video_shower_camara.stopped or video_getter.stopped:  # exit on ESC or q
    #if video_shower_MotionDetection.stopped or video_getter.stopped:  # exit on ESC or q
        video_getter.stop()
        video_shower_camara.stop()
        break


    video_shower_camara.frame = frame
