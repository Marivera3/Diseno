#! /bin/sh


 python3 extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 \
 	--shape-predictor shape_predictor_68_face_landmarks.dat
