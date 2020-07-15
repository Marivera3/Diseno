# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoGet.py
import threading
import cv2 as cv
import imutils
from imutils.video import FPS

class VideoGet(threading.Thread):

    def __init__(self,  frame_queue, src=0, name='Video Getter'):
        super().__init__(name=name)
        self.stream = cv.VideoCapture(src)
        (self.rval, self.frame) = self.stream.read()
        self.fps = FPS().start()
        self.stopped = False
        self.frame_queue = frame_queue


    def run(self):
        self.stopped = False
        print('[INFO] Thread started: {}...'.format(threading.current_thread().name))
        while not self.stopped:
            if self.rval:
                (self.rval, self.frame) = self.stream.read()
                # self.frame_queue.put(self.frame)
                self.fps.update()
            else:
                self.stop()
        print(f'[INFO] Releasing video capture...')
        self.stream.release()

    def stop(self):
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

        self.stopped = True
        self.rval = False

        # self.stream.release()
