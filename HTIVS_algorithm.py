# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:19:26 2019

@author: vildeg
"""

import cv2
import numpy as np
from functions import org_voting_scheme
from functions import draw_result_line
from functions import show_image
from functions import draw_hough_lines


def HTIVS_alg(img):
    edges = cv2.Canny(img,350,110)
    minLineLength = 10
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    a,b,rho,theta = org_voting_scheme(lines,90)
    return a,b,rho,theta
    
def draw_HTIVS_results(a,b,img):
    img_lined = draw_result_line(a,b,img,0,0,255)
    return img_lined