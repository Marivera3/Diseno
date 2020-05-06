# https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np
from showmotion import showmotion
from CountsPerSec import CountsPerSec

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


face_cascade = cv.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

detectmov = True
square = False

cv.namedWindow("Camera") # Creates a window
#cv.namedWindow("Motion Detection")
if square:
    cv.namedWindow("Motion")

# select the model of BS
backSub = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=50) # Parameters: history, varThreshold, bShadowDetection
# backSub = cv.createBackgroundSubtractorKNN()
vc = cv.VideoCapture(0)
cps = CountsPerSec().start()


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:


    fgmask = backSub.apply(frame)

    if detectmov:
        # mov: Treu if movement is detected; nomd: number of motion detected
        fmotion, mov = showmotion(fgmask.copy(), frame.copy(), square = square)
        if square:
            cv.imshow('Motion', fmotion)

        if mov:
            # print('processing faces...')
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5) # depend on the size of the image
            # faces is a list of list that each sub list give the pixel position of the edges of the frame
            [cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) for (x,y,w,h) in faces]
        else:
            print('no movements')
    frame = putIterationsPerSec(frame, cps.countsPerSec())
    cv.imshow('Camera', frame)
    #cv.imshow('Motion Detection', fgmask)

    key = cv.waitKey(1)
    if key == ord('q') or key == 27: # exit on ESC or q
        break
    cps.increment()
    rval, frame = vc.read()

vc.release()
cv.destroyWindow("Camera")
