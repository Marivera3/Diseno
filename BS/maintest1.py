# https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np
from showmotion import showmotion

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

face_cascade = cv.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

detectmov = True

cv.namedWindow("Camera") # Creates a window
cv.namedWindow("Motion Detection")
cv.namedWindow("Motion")

# select the model of BS
backSub = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=50) # Parameters: history, varThreshold, bShadowDetection
# backSub = cv.createBackgroundSubtractorKNN()
vc = cv.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:


    fgmask = backSub.apply(frame)

    if detectmov:
        # mov: Treu if movement is detected; nomd: number of motion detected
        fmotion, mov = showmotion(fgmask.copy(), frame.copy())

        if mov:
            print('processing faces...')
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5) # depend on the size of the image
            # faces is a list of list that each sub list give the pixel position of the edges of the frame
            [cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) for (x,y,w,h) in faces]
        else:
            print('no movements')    
        cv.imshow('Motion', fmotion)

    cv.imshow('Camera', frame)
    cv.imshow('Motion Detection', fgmask)

    key = cv.waitKey(20)
    if key == ord('q') or key == 27: # exit on ESC or q
        break

    rval, frame = vc.read()

vc.release()
cv.destroyWindow("Camera")
