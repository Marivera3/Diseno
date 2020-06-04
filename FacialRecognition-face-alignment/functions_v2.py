# USAGE
'''
python3 functions.py --detector face_detection_model \
--embedding-model openface_nn4.small2.v1.t7 \
--embeddings output/embeddings.pickle
'''

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


def rec_frame(detector, embedder, recognizer, le, vs):
    output = []
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            if proba > 0.65:
                name = le.classes_[j]
            else:
                name = 'unknown'

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            ## OUTPUTS
            output.append((name, vec))

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    return output

def rec_unknown(detector, embedder, recognizer, le, vs):
    output = []
    frame = vs.read()

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)

            output.append((frame, vec))

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    return output

def train(data):
    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    return recognizer, le

def change_name(name,id_unknown, embeddings):
    i=0
    for name in embeddings['names']:
        if name==id_unknown:
            embeddings['names'][i]=name
        i+=1
    print(dic)

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", required=True,
                    help="path to OpenCV's deep learning face detector")
    ap.add_argument("-e", "--embeddings", required=True,
                    help="path to serialized db of facial embeddings")
    ap.add_argument("-m", "--embedding-model", required=True,
                    help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(args["embeddings"], "rb").read())


    recognizer, le = train(data)

    # initialize the video stream, then allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)


    counter = 0
    new_unknown = 1

    counter_unknown = 1
    while True:
        output = rec_frame(detector, embedder, recognizer, le, vs)
        for item in output:
            name = item[0]
            if item[0] == 'unknown':
                c = 0
                prev_embeddings = []
                #Rectificar
                for i in range(20):
                    output = rec_frame(detector, embedder, recognizer, le, vs)
                    if item[0] == 'unknown':
                        c += 1
                    if not output:
                        continue
                    prev_embeddings.append(output[0][1].flatten())
                if c != 20:
                    continue
                nn = str(name) + ' {}'.format(counter_unknown)
                print(counter_unknown)
                data['embeddings'].append(item[1].flatten())
                data['names'].append(nn)
                for emb in prev_embeddings:
                    data['embeddings'].append(emb.flatten())
                    data['names'].append(nn)
                recognizer, le = train(data)
                for i in range(6):
                    output = rec_unknown(detector, embedder, recognizer, le, vs)
                    if not output:
                        continue
                    data['embeddings'].append(output[0][1].flatten())
                    data['names'].append(nn)
                recognizer, le = train(data)
                counter_unknown += 1
            else:
                if data['names'].count(item[0])<= 60:
                    data['embeddings'].append(item[1].flatten())
                    data['names'].append(str(item[0]))
                    recognizer, le = train(data)