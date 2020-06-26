import threading
import cv2
import datetime
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

        self.process        = Popen(['ffmpeg', '-y', '-f', 'image2pipe',
                                '-vcodec', 'mjpeg', '-loglevel', 'error',
                                '-use_wallclock_as_timestamps','1',
                                '-i', '-', '-vcodec', 'mpeg4', '-pix_fmt', 'yuv420p',
                                '-r', str(fps), pathOut], stdin=PIPE)

    def run(self):
        while not self.stopped and self.vg.rval:
            data = cv2.imencode('.jpg', self.vg.frame, self.encode_param)[1].tobytes()
            self.process.stdin.write(data)

        # self.process.stdin.close()
        # print(f'[INFO] Saving Video to {self.pathOut}...')


    def stop(self):
        self.stopped = True
        print(f'[INFO] Saving Video to {self.pathOut}...')
        self.process.stdin.close()
