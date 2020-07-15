# https://github.com/jrosebr1/imutils/blob/master/imutils/video/fps.py
import datetime

class FPS:
	def __init__(self, setFrames=2.0):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
		self.actualtime = None
		self._setFrames = setFrames
		self.boolFrames = True

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		self.actualtime = self._start
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

	def updateactualtime(self):
		self.actualtime = datetime.datetime.now()

	def istime(self):
		if ( datetime.datetime.now() - self.actualtime).total_seconds() > 1/self._setFrames:
			self.boolFrames = True
		else:
			self.boolFrames = False
