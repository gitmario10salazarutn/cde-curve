# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:51:46 2023

@author: Mario
"""

import cv2
import mediapipe as mp
import dlib
import math
import numpy as np

list_points = [127, 227, 93, 213, 138, 136, 149, 148, 152, 377, 378, 365, 367, 433, 352, 454, 368, 70, 63, 105, 65, 55, 285, 295, 334, 293, 300, 168, 197, 5, 4, 75, 97, 2, 326, 305, 246, 160, 158, 133, 153, 144, 362, 385, 387, 249, 373, 380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 96, 87, 14, 317, 307, 317, 14, 87]

s = "./img/angry.jpg"

# ************************************** Dlib Model 68 facial landmarks ***************************************************************
def dlib_model(image, y_pred):
    img_dlib = cv2.imread(image)
    # Detector facial
    face_detector = dlib.get_frontal_face_detector()
    # Predictor de 68 puntos de referencia
    predictor = dlib.shape_predictor("./face_detectors/shape_predictor_68_face_landmarks.dat")
    frame = img_dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coordinates_bboxes = face_detector(gray, 1)
    #print("coordinates_bboxes:", coordinates_bboxes)
    for c in coordinates_bboxes:
        x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
        cv2.rectangle(frame, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 1)
        shape = predictor(gray, c)
        for i in range(0, 68):
            x, y = shape.part(i).x, shape.part(i).y
            y_pred = np.append(y_pred, [int(x), int(y)])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(frame, str(i + 1), (x, y -5), 1, 0.8, (0, 255, 255), 1)
    return frame, y_pred


# ************************************** Mediapipe Model *****************************************************************************


def mediapipe_model(image, y_pred):
    cap = cv2.imread(image)
    height,width, _ = cap.shape
    mpDraw = mp.solutions.drawing_utils
    confDraw = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)
    mpMallaFacial = mp.solutions.face_mesh
    MallaFacial = mpMallaFacial.FaceMesh(max_num_faces = 1)
    while True:
        frame = cap
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = MallaFacial.process(frameRGB)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                 for index in list_points:
                    x = float(face_landmarks.landmark[index].x*width)
                    y = float(face_landmarks.landmark[index].y*height)
                    y_pred = np.append(y_pred, [int(x), int(y)])
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 255), 2)
                    cv2.putText(frame, str(index+1), (int(x), int(y)), 1, 0.8, (0, 255, 255), 1)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        else:
            break
    return frame, y_pred


def two_points_distance(pointa,pointb):
    first_term = pointa[0] - pointb[0]
    second_term = pointa[1] - pointb[1]
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))
y1 = []
y2 = []
f, y =dlib_model(s, y1)
f, x = mediapipe_model(s, y2)
cv2.imshow("Image", f)
cv2.waitKey(0)
cv2.destroyAllWindows()