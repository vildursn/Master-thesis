# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:59:13 2019

@author: vildeg
"""

import numpy as np
import cv2
from functions import draw_hough_lines
from functions import three_line_RANSAC
from functions import find_cart_line_eq
from functions import perpendicular_polar_line
from functions import lines_approx_parallel

from RANSAC_algorithm import RANSAC_alg 

def AMA_RANSAC_alg(img):
    edges = cv2.Canny(img,350,110)
    minLineLength = 10
    maxLineGap = 8
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    binary_img = np.zeros(np.shape(img))
    binary_img = draw_hough_lines(lines,binary_img,1)
    
    detected_lines_are_parallel = False
    times_tried = 0
    while detected_lines_are_parallel == False:
        
        max_attempts = 50+ times_tried*100
        detected_lines= three_line_RANSAC(binary_img,max_attempts)
        if len(detected_lines) == 0:
            print("no detected edges")
            break
            
        a1,b1,length1 = find_cart_line_eq(detected_lines[0][0],detected_lines[0][1],detected_lines[0][2],detected_lines[0][3])
        rho1, theta1, length1 = perpendicular_polar_line(a1,b1,length1)
            
        a2,b2,length2 = find_cart_line_eq(detected_lines[1][0],detected_lines[1][1],detected_lines[1][2],detected_lines[1][3])
        rho2, theta2, length2 = perpendicular_polar_line(a2,b2,length2)
        
        
        crossover_bool = True
        radians_allowed =np.pi/3
        img_heigth = np.shape(img)[0]
        if lines_approx_parallel(theta1,theta2,a1,b1,a2,b2,img_heigth,crossover_bool,radians_allowed):
            a3,b3,length3 = find_cart_line_eq(detected_lines[2][0],detected_lines[2][1],detected_lines[2][2],detected_lines[2][3])
            rho3, theta3, length3 = perpendicular_polar_line(a3,b3,length3)
            if (lines_approx_parallel(theta1,theta3,a1,b1,a3,b3,img_heigth,crossover_bool,radians_allowed) and lines_approx_parallel(theta2,theta3,a2,b2,a3,b3,img_heigth,crossover_bool,radians_allowed)):
                
                detected_lines_are_parallel = True
    
        times_tried += 1
        print(times_tried)
        if times_tried == 5 :
            detected_lines = RANSAC_alg(img)
            detected_lines_are_parallel = True
    return detected_lines

def draw_AMA_RANSAC_results(detected_lines, img):
    
    for i in range(0,len(detected_lines)):
        if i == 0:
            img_lined = cv2.line(img,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(255,0,0),2)
        if i == 1:
            img_lined = cv2.line(img,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,255,0),2)
        if i == 2:
            img_lined = cv2.line(img,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,0,255),2)
            
    return img_lined