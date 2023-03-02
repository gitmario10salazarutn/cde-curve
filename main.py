# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:43:02 2023

@author: Mario
"""

from models import dlib_model_68, mediapipe_model
from utils import find_face
import os
import numpy as np
import cv2


path_images = './Helen_testset/y_pred/'
path_pts = './Helen_testset/y_true/'
images = os.listdir(path_images)
images.sort()
pts = os.listdir(path_pts)
pts.sort()
print(len(images), len(pts))

a = "./img/a.jpg"
#dlib_model_68.dlib_model(a)
y_pred = []
#mediapipe_model.MediapipeModel(a, y_pred)
i = 315
image = path_images+images[i]
g = pts[i]
grand_truth = find_face.read_pts(g)

print(g)

find_face.grabcut(image, grand_truth)