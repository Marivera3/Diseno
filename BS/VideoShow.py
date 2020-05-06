import threading
import cv2 as cv
from CountsPerSec import CountsPerSec



class VideoShow:

    def __init__(self, frame=None, window=None):
        self.frame = frame
        self.window = window
        self.stopped = False
        self.mainthread = None
        self.cps = CountsPerSec()
        #cv.namedWindow(self.window)

    def show(self):
        print('Current Thread started: {}'.format(
            threading.current_thread().name))
        while not self.stopped:
            self.putIterationsPerSec()
            cv.imshow(self.window, self.frame)
            self.cps.increment()
            key = cv.waitKey(1)
            if key == ord('q') or key == 27:
                self.stopped = True

    def stop(self):
        self.stopped = True
        # cv.DestroyWindow(self.window)

    def start(self):
        self.cps.start()
        self.mainthread = threading.Thread(name='Video show {}'.format(
            self.window), target=self.show, args=()).start()
        return self


    def putIterationsPerSec(self):
        """
        Add iterations per second text to lower-left corner of a frame.
        """

        cv.putText(self.frame, "{:.0f} iterations/sec".format(self.cps.countsPerSec()),
            (10, 450), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
