# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:51:07 2023

@author: Mario
"""

import cv2
import dlib
import imutils
import math
import numpy as np
import PIL
from utils import find_face

def two_points_distance(x_ini, y_ini, x_fin, y_fin):
    first_term = x_ini - x_fin
    second_term = y_ini - y_fin
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))


def calculate_DF(width_face, height_face):
    return math.sqrt(width_face*height_face)

my_points = []

#a = "./img/image.jpg"

def dlib_model_98(grand_thrut, imagen, y_pred):
    index_list = [1, 3, 5, 7, 8, 10, 12, 14, 16, 18, 19, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 66, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 89, 92, 92, 93, 93, 94, 94]
    """
    
    [2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 20, 23, 25, 27, 29, 31, 33, 34, 35, 36, 37, 38, 43, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 67, 68, 69, 70, 71, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 90, 93, 93, 94, 94, 95, 95]
    
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 23, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 89, 91, 91, 92, 94, 94]
    """
    #cap = cv2.VideoCapture("video_Trim.mp4");
    #n = PIL.Image.open(img)
    # fetching the dimensions
    #wid, hgt = n.size
    frame = find_face.grabcut(imagen, grand_thrut)
    y_pred = []
    face_detector = dlib.get_frontal_face_detector()
    # Predictor de 68 puntos de referencia
    predictor = dlib.shape_predictor("./face_detectors/wflw_98_landmarks.dat")
    while True:
        #frame = imutils.resize(frame, width=wid, height=hgt)
        #print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        coordinates_boxes = face_detector(gray, 1)
        #print("coordinates_boxes: ", coordinates_boxes)
        
        for c in coordinates_boxes:
            #print(c)
            x_init, y_init, x_end, y_end = c.left(), c.top(), c.right(), c.bottom()
            cv2.rectangle(frame, (x_init, y_init), (x_end, y_end), (255, 255, 0), 1)
            shape = predictor(gray, c)
            for i in index_list:
                x, y = shape.part(i).x, shape.part(i).y
                my_points.append([x, y])
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

