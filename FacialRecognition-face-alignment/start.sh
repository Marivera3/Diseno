#!/bin/sh
python3 main_pi_v3.py --detector face_detection_model \
--embedding-model openface_nn4.small2.v1.t7 \
--embeddings output/embeddings.pickle
