# https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np

cv.namedWindow("Camera") # Creates a window

# select the model of BS
backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()
vc = cv.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()

    fgMask = backSub.apply(frame)
    print(fgMask.shape)

    #cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(frame, str(vc.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))


    cv.imshow('Camera', frame)
    cv.imshow('Motion Detection', fgMask)

    key = cv.waitKey(20)
    if key == ord('q') or key == 27: # exit on ESC or q
        break

vc.release()
cv.destroyWindow("Camera")
