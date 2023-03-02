# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:52:32 2023

@author: Mario
"""

''' 
Facial Landmark Detection in Python with OpenCV

'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np
import math
import dlib
from utils import find_face

def two_points_distance(x_ini, y_ini, x_fin, y_fin):
    first_term = x_ini - x_fin
    second_term = y_ini - y_fin
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))


def calculate_DF(width_face, height_face):
    return math.sqrt(width_face*height_face)


def HaarCascadeModel(grand_thrut, imagen, y_pred):
    y_pred = []
    # save face detection algorithm's url in haarcascade_url variable
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"
    haarcascade_clf = "data/" + haarcascade
    # check if data folder is in working directory
    if (os.path.isdir('data')):
        # check if haarcascade is in data directory
        if (haarcascade in os.listdir('data')):
            print("File exists")
        else:
            # download file from url and save locally as haarcascade_frontalface_alt2.xml
            urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
            print("File downloaded")
    else:
        # create data folder in current directory
        os.mkdir('data')
        # download haarcascade to data folder
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("File downloaded")
    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade_clf)
    # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "LFBmodel.yaml"
    LBFmodel_file = "data/" + LBFmodel
    # check if data folder is in working directory
    if (os.path.isdir('data')):
        # check if Landmark detection model is in data directory
        if (LBFmodel in os.listdir('data')):
            print("File exists")
        else:
            # download file from url and save locally as haarcascade_frontalface_alt2.xml
            urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
            print("File downloaded")
    else:
        # create data folder in current directory
        os.mkdir('data')
        # download Landmark detection model to data folder
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("File downloaded")
    # create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel_file)
    # get image from webcam
    print ("checking webcam for connection ...")
    #webcam_cap = cv2.imread(a)
    while(True):
        # read webcam
        frame = find_face.grabcut(imagen, grand_thrut)
        height, width, _ = frame.shape
        
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
        #print(len(faces))
        # Detect faces using the haarcascade classifier on the "grayscale image"
        #faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(int(height/2), int(350/2)),
		#flags=cv2.CASCADE_SCALE_IMAGE)
        """
        m = 500
        n =470
        faces = []
        print(height, width, len(faces), faces)
        while len(faces)==0:
            faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(int(m), int(m)), flags=cv2.CASCADE_SCALE_IMAGE)
            m-=10
            n-=10
        """
        
        if len(faces)==0:
            faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(200, 200), flags=cv2.CASCADE_SCALE_IMAGE)
            print(height, width, len(faces), faces, "1")
            if len(faces) == 0:
                 faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
                 print(height, width, len(faces), faces, "2")
                 if len(faces) == 0:
                    faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
                    print(height, width, len(faces), faces, "3")
                    if len(faces) == 0:
                        faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                        print(height, width, len(faces), faces, "4")
                        if len(faces) == 0:
                            faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
                            print(height, width, len(faces), faces, "5")
        
        #print(len(faces))
        face_detector = dlib.get_frontal_face_detector()
        """
        if width<1000:
            frame = imutils.resize(frame,width=1000, height=1000)
        else:
            frame = imutils.resize(frame,width=width, height=height)
        """
        gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        coordinates_bboxes= face_detector(gray,1)
        for c in coordinates_bboxes:
            x_init,y_init,x_end,y_end = c.left(),c.top(),c.right(),c.bottom()
            print("Hola Mario: ",x_init,y_init,x_end,y_end)
            break
            cv2.rectangle(frame, (x_init, y_init), (x_end, y_end), (255, 255, 0), 1)
        for (x_init1, y_init1, x_end1, y_end1) in faces:
            # Detect landmarks on "gray"
            z, landmarks = landmark_detector.fit(gray, np.array(faces))
            for landmark in landmarks:
                #print(landmark)
                i = 0
                for x , y in landmark[0]:
                    y_pred  = np.append(y_pred, [x, y])
                    # display landmarks on "frame/image,"
                    # with blue colour in BGR and thickness 2
                    cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)
                    cv2.putText(frame, str(i+1), (int(x), int(y) -5), 1, 0.8, (0, 255, 255), 1)
                    i+=1
                width_face = two_points_distance(x_init, y_init, x_end ,y_end )
                height_face = two_points_distance(x_init, y_init, x_end ,y_end)
                Df = calculate_DF(width_face, height_face)
                y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))
        # Show image
        """
        cv2.imshow("frame", frame)

        # terminate the capture window
        if cv2.waitKey(0) & 0xFF  == ord('q'):
            break
        else:
            cv2.destroyAllWindows()
            break
        """
        return y_pred, Df
        
#y, df = HaasCascadeModel(ia)

#print(y)
#print(df)