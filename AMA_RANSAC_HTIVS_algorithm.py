# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:02:14 2019

@author: vildeg
"""

import numpy as np
import cv2
from functions import draw_hough_lines
from functions import voting_scheme
from functions import three_line_RANSAC
from functions import find_cart_line_eq
from functions import perpendicular_polar_line
from functions import lines_approx_parallel
from functions import lines_result_approx_parallel
from functions import show_image
from functions import draw_result_line
from skimage.measure import ransac, LineModelND
import time

from functions import labeled_voting_scheme


def find_mid_line(power_lines):
    a1 = power_lines[0][0]
    a2 = power_lines[1][0]
    a3 = power_lines[2][0]
    b1 = power_lines[0][1]
    b2 = power_lines[1][1]
    b3 = power_lines[2][1]
    
    x1 = (500-b1)/a1
    x2 = (500-b2)/a2
    x3 = (500-b3)/a3
    if (x2 <= x1 and x1 <= x3):
        return 1,0,2
    if (x3 <= x1 and x1 <= x2):
        return 2,0,1
    if ( x1 <= x2 and x2 <= x3):
        return 0,1,2
    if (x3 <= x2 and x2 <=x1):
        return 2,1,0
    if (x1 <= x3 and x3 <= x2 ):
        return 0,2,1
    if (x2 <= x3 and x3 <= x1):
        return 1,2,0
    #if (b2 <= b1 and b1 <= b3) or (b3 <= b1 and b1 <= b2):
     #   return 0
    #if ( b1 <= b2 and b2 <= b3) or (b3 <= b2 and b2 <=b1):
     #   return 1
    #if (b1 <= b3 and b3 <= b2 ) or (b2 <= b3 and b3 <= b1):
       # return 2
       
def line_within_reach(a,b,x1,y1,x2,y2):
    length_accepted = 50
    if a == np.inf or a == 0:
        if (x1 < x2):
            X = b
            if (x1 > X-length_accepted ) and (x2 < X+length_accepted ):
                return True
            else:
                return False
        else:
            X = b
            if x2> X-length_accepted  and x1 < X+length_accepted :
                return True
            else:
                return False
        #hmmm
    if (x1 < x2):
        X = int((y1-b)/a)
        if (x1 > X-length_accepted ) and (x2 < X+length_accepted ):
            return True
        else:
            return False
    else:
        X = int((y2-b)/a)
        if x2> X-length_accepted  and x1 < X+length_accepted :
            return True
        else:
            return False


        
        
def find_mid_line(power_lines):
    a1 = power_lines[0][0]
    a2 = power_lines[1][0]
    a3 = power_lines[2][0]
    b1 = power_lines[0][1]
    b2 = power_lines[1][1]
    b3 = power_lines[2][1]
    if a1 == np.inf or a1 == 0:
        x1 = b1
    else:
        x1 = (500-b1)/a1
    if a2 == np.inf or a2 ==0:
        x2=b2
    else:
        x2 = (500-b2)/a2
    if a3 == np.inf or a3 ==0:
        x3 =b3
    else:
        x3 = (500-b3)/a3
    
    if (x2 <= x1 and x1 <= x3):
        return 1,0,2
    if (x3 <= x1 and x1 <= x2):
        return 2,0,1
    if ( x1 <= x2 and x2 <= x3):
        return 0,1,2
    if (x3 <= x2 and x2 <=x1):
        return 2,1,0
    if (x1 <= x3 and x3 <= x2 ):
        return 0,2,1
    if (x2 <= x3 and x3 <= x1):
        return 1,2,0
        


def label_lines_org(lines, ransac_lines,left_x_bound, right_x_bound):
    left_lines=[]
    mid_lines=[]
    right_lines=[]
    other_lines=[]
    power_mast_lines =[]
    print(ransac_lines)
    left_idx, mid_idx, right_idx = find_mid_line(ransac_lines)
    crossover_bool = True
    radians_allowed = np.pi/4
    for i in range(0,len(lines)):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        a,b,length = find_cart_line_eq(x1,y1,x2,y2)
        rho, theta, length = perpendicular_polar_line(a,b,length)
        if (x1 < left_x_bound) or (x2 < left_x_bound) or (x1 > right_x_bound) or (x2 > right_x_bound):
            other_lines.append(lines[i][0])
            #print("Other line")
        else:
            #print("går forbiii")
            a_l,b_l,length = find_cart_line_eq(ransac_lines[left_idx][0],ransac_lines[left_idx][1],ransac_lines[left_idx][2],ransac_lines[left_idx][3])
            rho_l, theta_l, length = perpendicular_polar_line(a_l,b_l,length)
            close_bool = line_within_reach(a_l,b_l,x1,y1,x2,y2)
            parallel_bool = lines_approx_parallel(theta_l,theta,a_l,b_l,a,b,1920,crossover_bool,radians_allowed )
            if close_bool and parallel_bool:
                left_lines.append(lines[i][0])
                
            else:
                #print("not left because par,close : ", parallel_bool, close_bool)
                #a_m = ransac_lines[mid_idx][0]
                #b_m = ransac_lines[mid_idx][1]
                #theta_m = ransac_lines[mid_idx][3]
                a_m,b_m,length = find_cart_line_eq(ransac_lines[mid_idx][0],ransac_lines[mid_idx][1],ransac_lines[mid_idx][2],ransac_lines[mid_idx][3])
                rho_m, theta_m, length = perpendicular_polar_line(a_m,b_m,length)
                close_bool = line_within_reach(a_m,b_m,x1,y1,x2,y2)
                parallel_bool = lines_approx_parallel(theta_m,theta,a_m,b_m,a,b,1920,crossover_bool,radians_allowed )
                if close_bool and parallel_bool:
                    mid_lines.append(lines[i][0])
                    
                    #print("Accepted as mid line ",theta_m,theta,b_m,b)
                else:
                    #print("not mid because par,close : ", parallel_bool, close_bool)
                    #print("NOT accepted as mid line ",np.rad2deg(theta_m),np.rad2deg(theta),b_m,b)
                    a_r,b_r,length = find_cart_line_eq(ransac_lines[right_idx][0],ransac_lines[right_idx][1],ransac_lines[right_idx][2],ransac_lines[right_idx][3])
                    rho_r, theta_r, length = perpendicular_polar_line(a_r,b_r,length)
                    close_bool = line_within_reach(a_r,b_r,x1,y1,x2,y2)
                    parallel_bool = lines_approx_parallel(theta_r,theta,a_r,b_r,a,b,1920,crossover_bool,radians_allowed )
                    if close_bool and parallel_bool:
                        right_lines.append(lines[i][0])
                    else:
                        #print("not right because par,close : ", parallel_bool, close_bool)
                        power_mast_lines.append(lines[i][0])
                        #print("power mast line")
    print(len(power_mast_lines), " power mast lines ")               
    return left_lines,mid_lines,right_lines, other_lines 


def ThreeLineRANSAC(binary_img,img_org,first_attempt):
    detected_lines_are_parallel = False
    times_tried = 0
    
    while detected_lines_are_parallel == False:
        
        max_attempts = first_attempt+ times_tried*100
        detected_lines= three_line_RANSAC(binary_img,max_attempts)
        if len(detected_lines) == 0:
            print("no detected edges")
            break
        print("RANSAC lines - ",detected_lines)
        
        a1,b1,length1 = find_cart_line_eq(detected_lines[0][0],detected_lines[0][1],detected_lines[0][2],detected_lines[0][3])
        rho1, theta1, length1 = perpendicular_polar_line(a1,b1,length1)
            
        a2,b2,length2 = find_cart_line_eq(detected_lines[1][0],detected_lines[1][1],detected_lines[1][2],detected_lines[1][3])
        rho2, theta2, length2 = perpendicular_polar_line(a2,b2,length2)
            
        if lines_approx_parallel(theta1,theta2,a1,b1,a2,b2,1920,True,np.pi/6):
            a3,b3,length3 = find_cart_line_eq(detected_lines[2][0],detected_lines[2][1],detected_lines[2][2],detected_lines[2][3])
            rho3, theta3, length3 = perpendicular_polar_line(a3,b3,length3)
            if (lines_approx_parallel(theta3,theta1,a3,b3,a1,b1,1920,True,np.pi/6) and lines_approx_parallel(theta3,theta2,a3,b3,a2,b2,1920,True,np.pi/6)):
                detected_lines_are_parallel = True
        else:
            print("lines are not parallel")
        times_tried += 1
        print("Times tried - ", times_tried)
        #print(times_tried)
        if times_tried == 5 :
            print("PROBLEM PROBLEM PROBLEM!!!")
            a3,b3,length3 = find_cart_line_eq(detected_lines[2][0],detected_lines[2][1],detected_lines[2][2],detected_lines[2][3])
            rho3, theta3, length3 = perpendicular_polar_line(a3,b3,length3)
            img_lined = img_org
            detected_lines_are_parallel = True
           
    power_lines = np.array([[a1,b1,rho1,theta1,length1],[a2,b2,rho2,theta2,length2],[a3,b3,rho3,theta3,length3]])
    indx = find_mid_line(power_lines)
    for i in range(0,len(detected_lines)):
        if i == indx:
            img_lined = cv2.line(img_org,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,255,0),2)
        else:
            img_lined = cv2.line(img_org,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,0,255),2)
        
    return img_lined,power_lines

def power_lines_approx_parallel_123(detected_lines):
    parallel_bool_12 = True
    parallel_bool_13 = True
    parallel_bool_23= True
    crossover_bool = False
    radians_allowed = np.pi/3
    img_heigth = 1080
    if detected_lines[0][4] == True:
        a1,b1,length1 = find_cart_line_eq(detected_lines[0][0],detected_lines[0][1],detected_lines[0][2],detected_lines[0][3])
        rho1, theta1, length1 = perpendicular_polar_line(a1,b1,length1)
        
    if detected_lines[1][4] == True:    
        a2,b2,length2 = find_cart_line_eq(detected_lines[1][0],detected_lines[1][1],detected_lines[1][2],detected_lines[1][3])
        rho2, theta2, length2 = perpendicular_polar_line(a2,b2,length2)
        
    if detected_lines[2][4] == True:
        a3,b3,length3 = find_cart_line_eq(detected_lines[2][0],detected_lines[2][1],detected_lines[2][2],detected_lines[2][3])
        rho3, theta3, length3 = perpendicular_polar_line(a3,b3,length3)
    
    if detected_lines[0][4] and detected_lines[1][4]:
        
        if not lines_result_approx_parallel(theta1,theta2,a1,b1,a2,b2,img_heigth,crossover_bool ,radians_allowed):
            parallel_bool_12 = False
    if detected_lines[0][4] and detected_lines[2][4]:
        if not (lines_result_approx_parallel(theta3,theta1,a3,b3,a1,b1,img_heigth,crossover_bool,radians_allowed)):
            parallel_bool_13 = False
    if detected_lines[1][4] and detected_lines[2][4]:
        if not lines_result_approx_parallel(theta3,theta2,a3,b3,a2,b2,img_heigth,crossover_bool ,radians_allowed):
            parallel_bool_23= False
    return parallel_bool_12, parallel_bool_13, parallel_bool_23

def power_lines_approx_parallel(detected_lines):
    parallel_bool_12 = True
    parallel_bool_13 = True
    parallel_bool_23= True
    parallel_bool = True
    crossover_bool = True
    radians_allowed = np.pi/5
    img_heigth = 1080
    
    if detected_lines[0][4] and detected_lines[1][4]:
        
        if not lines_result_approx_parallel(detected_lines[0][3],detected_lines[1][3],detected_lines[0][0],detected_lines[0][1],detected_lines[1][0],detected_lines[1][1],img_heigth,crossover_bool ,radians_allowed):
            parallel_bool_12 = False
            parallel_bool = False
            
    if detected_lines[0][4] and detected_lines[2][4]:
        if not (lines_result_approx_parallel(detected_lines[0][3],detected_lines[2][3],detected_lines[0][0],detected_lines[0][1],detected_lines[2][0],detected_lines[2][1],img_heigth,crossover_bool,radians_allowed)):
            parallel_bool_13 = False
            parallel_bool = False
    if detected_lines[1][4] and detected_lines[2][4]:
        if not lines_result_approx_parallel(detected_lines[2][3],detected_lines[1][3],detected_lines[2][0],detected_lines[2][1],detected_lines[1][0],detected_lines[1][1],img_heigth,crossover_bool ,radians_allowed):
            parallel_bool_23= False
            parallel_bool = False
    return parallel_bool_12, parallel_bool_13, parallel_bool_23,parallel_bool 
        

def AMA_RANSAC_HTIVS_alg(img, ransac_model):
    #print("AJJAJAJJAJAJAJA")
    edges = cv2.Canny(img,350,110)
    minLineLength = 10
    maxLineGap = 8
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    if len(lines) == 0:
        print("No lines found.. Continues searching")
        

    #left_lines, mid_lines, right_lines, other_lines= label_lines_only_reach(lines, ransac_model)
    
    #-----
    left_x_bound = 2000
    right_x_bound = 0
    #index = find_mid_line(ransac_model)
    #print("---",np.shape(ransac_model), " - ", ransac_model[0][0], "--", ransac_model)
    for j in range(0,3):
        #x_top = -int(ransac_model[j][1]/ransac_model[j][0])
        #x_low = int((np.shape(img)[0]-ransac_model[j][1])/ransac_model[j][0])
        x_top = ransac_model[j][0]
        x_low = ransac_model[j][2]
        if x_low > right_x_bound:
            right_x_bound = x_low
            #print("yes", right_x_bound)
        if x_top > right_x_bound:
            right_x_bound = x_top
        if x_low < left_x_bound:
            left_x_bound= x_low
        if x_top < left_x_bound:
            left_x_bound = x_top
    if left_x_bound > 100:
        left_x_bound -=100
    if right_x_bound < np.shape(img)[1] - 100:
        right_x_bound +=100
    #print("when stop")
    #print(left_x_bound," and ", right_x_bound, " are the bounds, x top and x_low are ",x_top, " and ", x_low)
    
    #-----
    left_lines, mid_lines, right_lines, other_lines = label_lines_org(lines, ransac_model,left_x_bound, right_x_bound)
               
    left_bool = len(left_lines)>= 5
    mid_bool = len(mid_lines) >= 5
    right_bool = len(right_lines) >= 5
    print("[r,l,m] = [",right_bool,",",left_bool, ",",mid_bool,"] = [",len(right_lines),",",len(left_lines), ",",len(mid_lines),"] with total ", len(lines), " lines")
     
    power_lines=[]
    if ((right_bool or left_bool or mid_bool) == False):
        return power_lines, True
    else:  
        #print("!!!!!!!! YES ")
        if left_bool:
           #img_lined = draw_labeled_lines(left_lines,img_org,0)
           a_l,b_l,r_l,t_l=labeled_voting_scheme(left_lines,90)
           power_lines.append([a_l,b_l,r_l,t_l,left_bool])
           x_l = (1080-b_l)/a_l
        else:
            power_lines.append([0,0,0,0,left_bool])
        if mid_bool:
            #img_lined = draw_labeled_lines(mid_lines, img_lined,1)
            a_m,b_m,r_m,t_m=labeled_voting_scheme(mid_lines,90)
            power_lines.append([a_m,b_m,r_m,t_m,mid_bool])
            x_m = (1080-b_m)/a_m
        else:
            power_lines.append([0,0,0,0,mid_bool])
        if right_bool:
           #img_lined = draw_labeled_lines(right_lines,img_lined,2)
           a_r,b_r,r_r,t_r=labeled_voting_scheme(right_lines,90)
           power_lines.append([a_r,b_r,r_r,t_r,right_bool])
           x_r = (1080-b_r)/a_r
        else:
           power_lines.append([0,0,0,0,right_bool])
        
        parallel_bool_12, parallel_bool_13, parallel_bool_23,parallel_bool  = power_lines_approx_parallel(power_lines)
        update_Ransac_bool = parallel_bool
        if (parallel_bool_12+ parallel_bool_13+ parallel_bool_23 >=2 ):
            update_Ransac_bool = False
        else:
            update_Ransac_bool = True
        accepted_dist = 50
        if left_bool and mid_bool and x_l != np.inf and x_m != np.inf:
            if np.abs(x_l - x_m) < accepted_dist:
                if parallel_bool_13:
                    power_lines[1][4]= False
                else:
                    power_lines[0][4] = False
        if left_bool and right_bool and x_l != np.inf and x_r != np.inf :
            if np.abs(x_l - x_r) < accepted_dist:
                if parallel_bool_12:
                    power_lines[2][4] = False
                else:
                    power_lines[0][4] = False
        if right_bool and mid_bool and x_r != np.inf and x_m != np.inf :
            if np.abs(x_r - x_m) < accepted_dist:
                if parallel_bool_12:
                    power_lines[2][4] = False
                else:
                    power_lines[1][4] = False
                    
        #if (parallel_bool_12 + parallel_bool_13 <2):
         #   power_lines[0][4] = False
        #if (parallel_bool_12 + parallel_bool_23 <2):
         #   power_lines[1][4] = False
        #if (parallel_bool_13 + parallel_bool_23 <2):
         #   power_lines[2][4] = False
        return power_lines, update_Ransac_bool
    
def draw_AMA_RANSAC_HTIVS_results(img,power_lines):
    img_lined = img
    if power_lines[0][4] == True:
        a2,b2,length2 = find_cart_line_eq(power_lines[0][0],power_lines[0][1],power_lines[0][2],power_lines[0][3])
        rho2, theta2, length2 = perpendicular_polar_line(a2,b2,length2)
        print("LEFT LINE : ", np.rad2deg(theta2))
        img_lined = draw_result_line(power_lines[0][0],power_lines[0][1],img,0,0,255)
    if power_lines[1][4] == True:
        a2,b2,length2 = find_cart_line_eq(power_lines[1][0],power_lines[1][1],power_lines[1][2],power_lines[1][3])
        rho2, theta2, length2 = perpendicular_polar_line(a2,b2,length2)
        print("MID LINE : ", np.rad2deg(theta2))
        img_lined = draw_result_line(power_lines[1][0],power_lines[1][1],img,0,255,0)
    if power_lines[2][4] == True:
        a2,b2,length2 = find_cart_line_eq(power_lines[2][0],power_lines[2][1],power_lines[2][2],power_lines[2][3])
        rho2, theta2, length2 = perpendicular_polar_line(a2,b2,length2)
        print("RIGHT LINE : ", np.rad2deg(theta2))
        img_lined = draw_result_line(power_lines[2][0],power_lines[2][1],img,255,0,0)
    return img_lined
        