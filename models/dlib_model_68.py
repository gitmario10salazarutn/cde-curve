# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:51:25 2023

@author: Mario
"""

import cv2
import dlib
import imutils
import math
import numpy as np
import PIL
from PIL import Image
from utils import find_face

def two_points_distance(x_ini, y_ini, x_fin, y_fin):
    first_term = x_ini - x_fin
    second_term = y_ini - y_fin
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))


def calculate_DF(width_face, height_face):
    return math.sqrt(width_face*height_face)


#a = "./img/foto_005.jpg"
def dlib_model(grand_thrut, imagen, y_pred):
    #cap = cv2.VideoCapture("video_Trim.mp4");
    #n = PIL.Image.open(img)
    # fetching the dimensions
    #wid, hgt = n.size
    #frame = cv2.imread(img)
    frame = find_face.grabcut(imagen, grand_thrut)
    face_detector = dlib.get_frontal_face_detector()
    # Predictor de 68 puntos de referencia
    predictor = dlib.shape_predictor("./face_detectors/shape_predictor_68_face_landmarks.dat")
    while True:
        #frame = imutils.resize(frame, width=wid, height=hgt)
        #print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coordinates_boxes = face_detector(gray, 1)
        #print("coordinates_boxes: ", coordinates_boxes)
        
        for c in coordinates_boxes:
            x_init, y_init, x_end, y_end = c.left(), c.top(), c.right(), c.bottom()
            cv2.rectangle(frame, (x_init, y_init), (x_end, y_end), (255, 255, 0), 1)
            shape = predictor(gray, c)
            for i in range (0, 68):
                x, y = shape.part(i).x, shape.part(i).y
                y_pred= np.append(y_pred, [x, y])
                cv2.circle(frame, (x, y), 2, (0, 0, 255))
                cv2.putText(frame, str(i+1), (x, y -5), 1, 0.8, (0, 255, 255), 1)
            width_face = two_points_distance(x_init, y_init, x_end, y_end)
            height_face = two_points_distance(x_init, y_init, x_end, y_end)
            Df = calculate_DF(width_face, height_face)
            y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))
            #print(y_pred, Df, frame)
        #cv2.imshow("Picture", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        else:
            break
    return y_pred, Df
#y, df = dlib_model(a)
#print(y)
#print(df)
cv2.destroyAllWindows()

