import threading
import cv2
import datetime
import queue
import time
import multiprocessing as mp
from subprocess import Popen, PIPE

class SaveVideo(threading.Thread):

    def __init__(self, name, vg, pathOut, fps=25, encode_quality=95):
        # '-t', str(time)
        super().__init__(name=name)
        print('[INFO] Conexion for Video Creation using ffmpeg...')
        self.vg             = vg
        self.pathOut        = pathOut
        self.encode_param   = [cv2.IMWRITE_JPEG_QUALITY, encode_quality]
        self.stopped        = False
        #
        #
        self.process        = Popen(['ffmpeg',
                                    '-y', # overwrite output file
                                    '-async', '0',
                                    '-f', 'image2pipe',
                                    '-vcodec', 'mjpeg',
                                    '-use_wallclock_as_timestamps','1',
                                    '-i', '-', # The input comes from a pipe
                                    '-an', # Tells ffmpeg not to expect any audio
                                    '-loglevel', 'error',
                                    '-pix_fmt', 'yuv420p',
                                    '-vcodec', 'mpeg4',
                                    '-q','5', # Variable Bit Rate: number from 1-31, with 1 being highest quality/largest filesize and 31 being the lowest quality/smallest filesize
                                    '-r', str(fps),
                                    pathOut,
                                    '-async', '1',
                                    '-vsync', '1'
                                    ], stdin=PIPE)

    def run(self):
        c = 0
        while True:
            try:
                # print(f'Queue size: {self.vg.qsize()}')
                data = cv2.imencode('.jpg', self.vg.get(False), self.encode_param)[1].tostring()
                self.process.stdin.write(data)
                c = 0
            except queue.Empty:
                # Handle empty queue here
                c += 1
                time.sleep(c/10.0)
                print(f'c : {c}')
                if c > 10:
                    print('break cicle')
                    break


        print(f'[INFO] Saving Video to {self.pathOut}...')
        self.process.stdin.close()


    def stop(self):
        self.stopped = True
        print(f'[INFO] Saving Video to {self.pathOut}...')
        self.process.stdin.close()
