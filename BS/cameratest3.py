import cv2 as cv
import numpy as np

cv.namedWindow("Camera") # Creates a window

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

#face_cascade = cv.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml') # smaller size

# select the model of BS
# backSub = cv.createBackgroundSubtractorMOG2()
# backSub = cv.createBackgroundSubtractorKNN()
vc = cv.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()

    #fgMask = backSub.apply(frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # depend on the size of the image
    # faces is a list of list that each sub list give the pixel position of the edges of the frame
    [cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) for (x,y,w,h) in faces]

    #print(faces)
    cv.imshow('Camera', frame)


    key = cv.waitKey(20)
    if key == ord('q') or key == 27: # exit on ESC or q
        break

vc.release()
cv.destroyWindow("Camera")
