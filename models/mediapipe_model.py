# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:52:13 2023

@author: Mario
"""

import math
import mediapipe as mp
import cv2
import imutils
import PIL
import dlib
import numpy as np
from utils import find_face

def two_points_distance(x_ini, y_ini, x_fin, y_fin):
    first_term = x_ini - x_fin
    second_term = y_ini - y_fin
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))


def calculate_DF(width_face, height_face):
    return math.sqrt(width_face*height_face)


def MediapipeModel(grand_thrut, imagen, y_pred):
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    #mp_drawing = mp.solutions.drawing_utils
    index_list = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107, 336,
                          296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373,
                          380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5) as face_detection:
        #image = cv2.imread(imagen)
        image = find_face.grabcut(imagen, grand_thrut)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
    #print("Detections:", results.detections)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        n = PIL.Image.open(imagen)
        # fetching the dimensions
        wid, hgt = n.size
        #image = cv2.imread(imagen)
        image = find_face.grabcut(imagen, grand_thrut)
        height,width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        face_detector = dlib.get_frontal_face_detector()
        if width<1000:
            frame = imutils.resize(image,width=1000, height = height)
        else:
            frame = imutils.resize(image,width=width, height = height)
        gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        coordinates_bboxes= face_detector(gray,1)
        for c in coordinates_bboxes:
            x_init,y_init,x_end,y_end = c.left(),c.top(),c.right(),c.bottom()
            print("Hola Mario: ",x_init,y_init,x_end,y_end)
            
        #print("Face landmarks: ", results.multi_face_landmarks)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in index_list:
                    x = float(face_landmarks.landmark[index].x*width)
                    y = float(face_landmarks.landmark[index].y*height)
                    y_pred = np.append(y_pred, [int(x), int(y)])
                    cv2.circle(image, (int(x), int(y)), 2, (255, 0, 255), 2)
                    cv2.putText(image, str(index+1), (int(x), int(y)), 1, 0.8, (0, 255, 255), 1)
                    cv2.rectangle(image, (int(x_init/2), int(y_init/2)), (int(x_end/2), int(y_end/2)), (0, 255, 0), 2)
            width_face = two_points_distance(x_init, y_init, x_end, y_end)
            height_face = two_points_distance(x_init, y_init, x_end, y_end)
            Df = calculate_DF(width_face, height_face)
            y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))
            
                #cv2.imshow("Image", image)
                #cv2.waitKey(0)
        '''
        if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image, face_landmarks,
                        mp_face_mesh.FACE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=-1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))
        '''
        #cv2.imshow("Image", image)
        cv2.waitKey(0)
    return y_pred, Df
cv2.destroyAllWindows()
