# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:15:37 2019

@author: vildeg
"""
import numpy as np
import math
import cv2
from functions import draw_hough_lines
from functions import three_line_RANSAC
from functions import find_cart_line_eq
from functions import perpendicular_polar_line
from functions import lines_approx_parallel
from functions import show_image
from functions import draw_result_line
from sklearn.cluster import KMeans
from functions import lines_approx_parallel
import matplotlib.pyplot as plt
from functions import three_line_RANSAC
from functions import kmeans_voting_scheme
from functions import sort_lines


def KMeans_alg(img, power_lines):
    edges = cv2.Canny(img,350,110)
    minLineLength = 10
    maxLineGap = 8
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    polar_lines=[]
    cartesian_lines=[]
    lengths = []
    max_rho = 0
    min_rho= 0
    for i in range(0,len(lines)):
        a,b,length = find_cart_line_eq(lines[i][0][0],lines[i][0][1],lines[i][0][2],lines[i][0][3])
        rho, theta, length = perpendicular_polar_line(a,b,length)
        if rho < min_rho:
            min_rho = rho
        if rho > max_rho :
            max_rho = rho
        polar_lines.append([rho,theta])
        cartesian_lines.append([lines[i][0][0],lines[i][0][1],lines[i][0][2],lines[i][0][3]])
        lengths.append(length)
    
    scaled_polar_lines = polar_lines
    for i in range(0,len(scaled_polar_lines)):
        scaled_polar_lines[i][0] = 180*(scaled_polar_lines[i][0] - min_rho)/(max_rho - min_rho)
        #scaled_polar_lines[i][0] = 0
        scaled_polar_lines[i][1] = scaled_polar_lines[i][1]
           
        
    n_clusters = 3
    kmeans = KMeans(n_clusters=3, random_state=0).fit(scaled_polar_lines)
    clustering = kmeans.predict(scaled_polar_lines)
    votes = np.zeros(n_clusters)
    for i in range(0,len(lines)):
           #img_hough =  cv2.line(img,(lines[i][0],lines[i][1]),(lines[i][2],lines[i][3]),(a*255,255,0),2)
            for x1,y1,x2,y2 in lines[i]:
                a = clustering[i]
                if a == 0:
                    votes[0] += lengths[i]
                elif a == 1:
                    votes[1] +=lengths[i]
                else: 
                    votes[2] += lengths[i]
    ix = np.argmax(votes)
        
    sorted_lines,number_added = sort_lines(cartesian_lines,clustering)
    a,b,rho,theta = kmeans_voting_scheme(sorted_lines[ix],180, int(number_added[ix]))
        
        
    if lines_approx_parallel(power_lines[ix][3],theta,power_lines[ix][0],power_lines[ix][1],a,b,np.shape(img)[0],True,np.pi/2):
            #img_lined = draw_result_line(a,b,img_org,255,255,0)
        return a,b,False
    else:
        votes[ix] = 0
        ix = np.argmax(votes)
        a,b,rho,theta = kmeans_voting_scheme(sorted_lines[ix],180, int(number_added[ix]))
        if lines_approx_parallel(power_lines[ix][3],theta,power_lines[ix][0],power_lines[ix][1],a,b,np.shape(img)[0],True,np.pi/2):
            return a,b,False
                #img_lined = draw_result_line(a,b,img_org,255,255,0)
        else:
                #img_lined = draw_result_line(a,b,img_org,255,255,0)
            return 0,0, True
                #img_lined, power_lines,index = update_RANSAC(lines, img_org)
                
def draw_KMEANS_results(a,b,img):
    img_lined = draw_result_line(a,b,img,255,255,0)
    return img_lined