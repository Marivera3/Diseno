#!/bin/sh
python3 main_pi_v7.py --detector face_detection_model \
--embeddings output/embeddings.pickle \
--embedding-model openface_nn4.small2.v1.t7 \
--confidence 0.5 --shape-predictor shape_predictor_68_face_landmarks.dat
