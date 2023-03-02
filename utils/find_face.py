# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:18:57 2023

@author: Mario
"""

import cv2
import face_recognition
import numpy as np


# pip install face-recognition opencv-contrib-python


def quitar_ruido(imagen, det, vent):
    det = int(det)
    vent = int(vent)

    img_new = cv2.fastNlMeansDenoisingColored(imagen, None, det, det, vent,10)

    print("Imagen original")
    cv2.imshow("01", imagen)

    print("Imagen con filtro de ventana de tamaño", vent)
    cv2.imshow("02", img_new)

#quitar_ruido(img,1, 5)

def increase_image(x, y, w, h, percentage):
    x = x - percentage
    y = y - percentage
    w = w + percentage
    h = h + percentage
    return int(x), int(y), int(w), int(h)

def found_min_max_points(grand_truth):
    x = []
    y = []
    for coord_x, coord_y in grand_truth:
        x.append(coord_x)
        y.append(coord_y)
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def my_face_found_GrandTruth(grand_truth, image, percentage):
    try:
        y_true = []
        img = cv2.imread(image)
        xa, ya, wa, ha = found_min_max_points(grand_truth)
        x, y, w, h = increase_image(x = xa, y = ya, w = wa, h = ha, percentage=percentage)
        new_img = img[y:h, x:w]
        wid = new_img.shape[0]
        hgt = new_img.shape[1]
        new_img = cv2.resize(new_img, (wid, hgt))
        for x, y in grand_truth:
            cv2.circle(new_img, (int(x-wid+percentage), int(y-hgt+percentage)), 2, (0, 255, 8), 5)
            y_true.append([x-wid + percentage, y-hgt+percentage])
        #cv2.imshow("Crooped: ", new_img)
        return new_img, y_true
    except Exception as e:
        raise Exception(e)

def read_pts(filename):
  path_pts = './Helen_testset/y_true/'
  return np.loadtxt(path_pts + filename, comments=("version:", "n_points:", "{", "}"))

def grabcut(img, grand_truth):
    img = cv2.imread(img)
    seg = np.zeros(img.shape[:2],np.uint8)
    x, y, width, height = found_min_max_points(grand_truth)
    seg[y:y+height, x:x+width] = 1
    background_mdl = np.zeros((1,65), np.float64)
    foreground_mdl = np.zeros((1,65), np.float64)
    cv2.circle(img, (int(x), int(y)), 2, (0, 255, 8), 5)
    cv2.circle(img, (int(width), int(height)), 2, (0, 255, 8), 5)
    cv2.grabCut(img, seg, (x-100, y-100, width, height), background_mdl, foreground_mdl, 0,
    cv2.GC_INIT_WITH_RECT)
    mask_new = np.where((seg==2)|(seg==0),0,1).astype('uint8')
    img = img*mask_new[:,:,np.newaxis]
    #cv2.imshow('Output', img)
    cv2.waitKey(0)
    return img


cv2.waitKey(0)
cv2.destroyAllWindows()

