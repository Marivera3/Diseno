# https://github.com/nrsyed/computer-vision/blob/master/multithread/VideoGet.py
import threading
import cv2 as cv


class VideoGet:

    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        (self.rval, self.frame) = self.stream.read()
        self.stopped = False
        self.mainthread = None

    def get(self):
        print('Current Thread started: {}'.format(threading.current_thread().name))
        while not self.stopped:
            if self.rval:
                (self.rval, self.frame) = self.stream.read()
            else:
                self.stop()

    def stop(self):

        self.stopped = True
        self.stream.release()

    def start(self):
        self.stopped = False
        self.mainthread = threading.Thread(
            name='Thread Video', target=self.get, args=()).start()
        return self
