class TrackableObject:
	def __init__(self, objectID, centroid, rc):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		self.sent = False
		if rc != 'unknown':
			self.reconocido = True
		else:
			self.reconocido = False
