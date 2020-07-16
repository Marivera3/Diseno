import cv2
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from sklearn.manifold import TSNE
'''
python3 plot_emb.py --embeddings output/embeddings.pickle
'''

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
				help="path to serialized db of facial embeddings")
args = vars(ap.parse_args())


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())


X_embedded = TSNE(n_components=2).fit_transform(data['embeddings'])

#print(X_embedded)
plt.figure()
t = set(data['names'])
for i,ta in enumerate(t):
    idx = np.where(np.array(data['names'])==ta)
    print(X_embedded[idx, 0],X_embedded[idx, 1])
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=ta)

plt.legend(bbox_to_anchor=(1, 1))
plt.show()

plt.figure()
idx = np.where(np.array(data['names'])=='keanu_reeves')
plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label='keanu_reeves')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()
