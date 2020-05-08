# Facial Recognition and Movement Detection

The BS folder contain some script test for Facial Recognition and Movement Detection. Also have a file called requirements.txt, that is used to set the venv.

These tasks are based on OpenCV2 library.

The codes are

- [cameratest.py](https://github.com/Marivera3/Diseno/blob/master/BS/cameratest.py): It is for know how to use the camere with the library.
- [camperatest2.py](https://github.com/Marivera3/Diseno/blob/master/BS/cameratest2.py): Show the detection movement based on Background Substrasting.
- [cameratest3.py](https://github.com/Marivera3/Diseno/blob/master/BS/cameratest3.py): Show the facial recognition using the HaarCascade algorithm.
- [maintest1.py](https://github.com/Marivera3/Diseno/blob/master/BS/maintest1.py): Mixed the last 2 codes and only detects faces when detect movements.
- [showmotion.py](https://github.com/Marivera3/Diseno/blob/master/BS/showmotion.py): Add blue square to the pixels that moves.


For Haar Cascade you can search here:
https://github.com/Itseez/opencv/tree/master/data/haarcascades

Some usefull information of the HaarCascade:
https://www.researchgate.net/publication/3940582_Rapid_Object_Detection_using_a_Boosted_Cascade_of_Simple_Features
https://docs.opencv.org/4.0.0/d7/d8b/tutorial_py_face_detection.html



In the folder FacialDetection is an optimized code. Here we use an OpenCV’s deep learning face detector,based on the Single Shot Detector (SSD) framework with a ResNet base network. Implemented in Caffe.


