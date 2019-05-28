# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:47:51 2019

@author: vildeg
"""
import numpy as np
import cv2
from functions import draw_hough_lines
from functions import three_line_RANSAC
from skimage.measure import ransac, LineModelND

def RANSAC_alg(img):
    edges = cv2.Canny(img,350,110)
    minLineLength = 10
    maxLineGap = 8
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    binary_img = np.zeros(np.shape(img))
    binary_img = draw_hough_lines(lines,binary_img,1)
    detected_lines= three_line_RANSAC(binary_img,1000)
    return detected_lines

def draw_RANSAC_results(detected_lines, img):
    
    for i in range(0,len(detected_lines)):
        if i == 0:
            img_lined = cv2.line(img,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(255,0,0),2)
        if i == 1:
            img_lined = cv2.line(img,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,255,0),2)
        if i == 2:
            img_lined = cv2.line(img,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,0,255),2)
            
    return img_lined