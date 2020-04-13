# https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
# https://www.programcreek.com/python/example/67891/cv2.imshow

import cv2
import numpy as np

cv2.namedWindow("Camera") # Creates a window
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    frame[:, :, 0] = 0.35*frame[:, :, 0];
    frame[:, :, 1] = 0.45*frame[:, :, 1];
    frame[:, :, 2] = 0.1*frame[:, :, 2];
    cv2.imshow("Camera", frame)
    rval, frame = vc.read()
    # print(frame.shape) # (480, 640, 3)
    key = cv2.waitKey(20)
    if key == ord('q') or key == 27: # exit on ESC or q
        break

vc.release()
cv2.destroyWindow("Camera")
