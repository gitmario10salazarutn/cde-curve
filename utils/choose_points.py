# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:51:46 2023

@author: Mario
"""

import cv2
import mediapipe as mp
import dlib
import math


image = "../img/a.jpg"

# ************************************** Dlib Model 68 facial landmarks ***************************************************************

x_y_dlib = [None]*68

img_dlib = cv2.imread(image)
# Detector facial
face_detector = dlib.get_frontal_face_detector()
# Predictor de 68 puntos de referencia
predictor = dlib.shape_predictor("../face_detectors/shape_predictor_68_face_landmarks.dat")
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
        x_y_dlib[i] = ((x, y), i+1)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        #cv2.putText(frame, str(i + 1), (x, y -5), 1, 0.8, (0, 255, 255), 1)
#cv2.imshow("Frame", frame)
#print(x_y_dlib)

# ************************************** Mediapipe Model *****************************************************************************

x_y_mediapipe = [None]*468
cap = cv2.imread(image)
mpDraw = mp.solutions.drawing_utils
confDraw = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces = 1)
while True:
    frame = cap
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = MallaFacial.process(frameRGB)
    if results.multi_face_landmarks:
        for rostros in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, confDraw, confDraw)
            for id, puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x, y = int(puntos.x*an), int(puntos.y*al)
                x_y_mediapipe[id]=((x, y), id+1)
                cv2.circle(frame, (x, y), 1, (255, 0, 255), 1)
                #cv2.putText(frame, str(id+1), (x, y), 1, 0.8, (0, 255, 255), 1)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    else:
        break
#print(x_y_mediapipe)
#cv2.imshow("IM", frame)
def two_points_distance(pointa,pointb):
    first_term = pointa[0] - pointb[0]
    second_term = pointa[1] - pointb[1]
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))

"""

#print(x_y_dlib)

A = [None]*68
z = 0
for pa, a in x_y_dlib:
    dist = []
    for pb, b in x_y_mediapipe:
        d = two_points_distance(pa, pb)
        dist.append((pb, b, d))
    A[z] = (pa, a, dist)
    z+=1

#print((A[3])[2])

#t = ((A[0])[2:])[0]
list_min = []
points_mins = []

for i in range(len(A)):
    t = ((A[i])[2:])[0]
    dis = []
    #print(t)
    for j in range(len(t)):
        dis.append((t[j])[2])
    dmin = 0
    d_min = min(dis)
    for k in range(len(t)):
        if d_min == (t[k])[2]:
            #print(d_min , (t[k])[2], k)
            points_mins.append((t[k]))
            break

print(len(points_mins))
o = []
for a, b, c in points_mins:
    #print(a, b, c)
    o.append(b)
o.sort()
print(o)
#cv2.imshow("Mediapipe Model 468 facial Landmarks", frame)

"""
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(x2_y2)
# NTCNN, dlib, mediapipe, TCNN


# 