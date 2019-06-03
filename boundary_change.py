# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:35:13 2019

@author: vildeg
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:44:55 2019

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
from functions import show_image
from functions import draw_result_line
from skimage.measure import ransac, LineModelND
from functions import labeled_voting_scheme

def length_of_line(x1,y1,x2,y2):
    length = np.sqrt(np.square(x2-x1)+np.square(y2-y1))
    return length
         
def power_mast_detector(power_mast_lines,theta_l,a_l,b_l,theta_m,a_m,b_m, theta_r,a_r,b_r, l_bool, m_bool, r_bool):
    accepted_lines =[]
    other_lines = []
    for i in range(0,len(power_mast_lines)): 
        a,b,length = find_cart_line_eq(power_mast_lines[i][0],power_mast_lines[i][1],power_mast_lines[i][2],power_mast_lines[i][3])
        rho, theta, length = perpendicular_polar_line(a,b,length)
        if lines_approx_orthogonal(theta,a,b,theta_l,a_l,b_l,theta_m,a_m,b_m, theta_r,a_r,b_r, l_bool, m_bool, r_bool):
            accepted_lines.append(power_mast_lines[i])
        else:
            other_lines.append(power_mast_lines[i])
            
    if len(accepted_lines) > 1:
        score = np.zeros(len(accepted_lines))
        
        for i in range(0, len(accepted_lines)):
            for j in range(0,len(accepted_lines)):
                if i != j:
                    if is_neighbour_line(accepted_lines[i][1],accepted_lines[i][3],accepted_lines[j][1],accepted_lines[j][3],5):
                        score[i] += length_of_line(accepted_lines[j][0],accepted_lines[j][1],accepted_lines[j][2],accepted_lines[j][3])
        core_line_indx = np.argmax(score)
        max_score = np.max(score)
        if max_score > 100:
            return True,accepted_lines,accepted_lines[core_line_indx]
        else:
            return False, accepted_lines, accepted_lines[core_line_indx]
    else:
        return False, accepted_lines, accepted_lines

def draw_outlines_power_masts(img,core_line, ransac_lines, left_idx, right_idx):
    y1=core_line[1]
    y2 = core_line[3]
    left_x = int((y1-ransac_lines[left_idx][1])/ransac_lines[left_idx][0])-20
    right_x = int((y2-ransac_lines[right_idx][1])/ransac_lines[right_idx][0])+20
    cv2.line(img,(left_x,int(core_line[1]+50)),(left_x,int(core_line[1]-100)),(255,100,255),3)
    cv2.line(img,(right_x,int(core_line[1]+50)),(right_x,int(core_line[1]-100)),(255,100,255),3)
    cv2.line(img,(left_x, int(core_line[1]+50)),(right_x, int(core_line[1]+50)),(255,100,255),3)
    cv2.line(img,(left_x, int(core_line[1]-100)),(right_x, int(core_line[1]-100)),(255,100,255),3)
    return img

def line_cross_bool(a1,b1,a2,b2):
    if a2 == a1:
        return False
    else:
        x = (b1-b2)/(a2-a1)
        y = a1*x+b1
        #print(y," > ", img_heigth/4, " and ",y," < " ,img_heigth)
        if ( y < 0 or y > 1080): #or y > (img_heigth - img_heigth/4)  np.isnan(y) or
            #print(y," > ", img_heigth/4, " and ",y," < " ,img_heigth)
            return False
        else:
            return True
    
def lines_approx_orthogonal(theta,a,b,theta_l,a_l,b_l,theta_m,a_m,b_m, theta_r,a_r,b_r, l_bool, m_bool, r_bool):
    diff_l = np.abs(theta-theta_l)
    diff_m = np.abs(theta-theta_m)
    diff_r=np.abs(theta -theta_r)
    ret_bool = False
    low_angle = 75
    high_angle = 105
    if (diff_l < np.deg2rad(high_angle)) and (diff_l > np.deg2rad(low_angle) or( diff_m < np.deg2rad(high_angle)) and (diff_m > np.deg2rad(low_angle))) or (diff_r < np.deg2rad(high_angle) and (diff_r > np.deg2rad(low_angle))):
        #print("True with difference : ", np.rad2deg(diff_m))
        ret_bool = True
    cross_l = line_cross_bool(a,b,a_l,b_l)
    cross_m = line_cross_bool(a,b,a_m,b_m)
    cross_r = line_cross_bool(a,b,a_r,b_r)
    
    if cross_l and cross_m and cross_r and ret_bool:#and ret_bool
        
        return True
    else:
       #print("Cross L,M,R = ", cross_l, cross_m, cross_r," and orthogonal : ", ret_bool)
        return False
    
    
    
def cross_bool(a1,b1,a2,b2,left_x_bound, right_x_bound):
    if a2 == a1:
        return False
    else:
        
        x = (b1-b2)/(a2-a1)
        y = a1*x+b1
        #print(y," > ", img_heigth/4, " and ",y," < " ,img_heigth)
        if ( y < 0 or y > 1080 or x <left_x_bound or x> right_x_bound) : #or y > (img_heigth - img_heigth/4)  np.isnan(y) or
            #print(y," > ", img_heigth/4, " and ",y," < " ,img_heigth)
            return False
        else:
            return True
            
def is_neighbour_line(y1,y2,Y1,Y2,accepted_length):
    if (np.abs(y1-Y1) < accepted_length) or (np.abs(y1-Y2) < accepted_length) or ( np.abs(y2-Y1) < accepted_length) or (np.abs(y2 - Y2) < accepted_length):
        return True
    else:
        return True

def power_lines_approx_parallel(detected_lines):
    parallel_bool = True
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
        
        if not lines_approx_parallel(theta1,theta2,a1,b1,a2,b2,1920,True,np.pi/2):
            parallel_bool = False
    if detected_lines[0][4] and detected_lines[2][4]:
        if not (lines_approx_parallel(theta3,theta1,a3,b3,a1,b1,1920,True,np.pi/2)):
            parallel_bool = False
    if detected_lines[1][4] and detected_lines[2][4]:
        if not lines_approx_parallel(theta3,theta2,a3,b3,a2,b2,1920,True,np.pi/2):
            parallel_bool= False
    #print("Power Lines Parallel : " ,parallel_bool)
    return parallel_bool

def line_within_reach(a,b,x1,y1,x2,y2):
    length_accepted = 20
    if a == np.inf:
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
    l_index,m_index,r_index = find_mid_line(power_lines)
    for i in range(0,len(detected_lines)):
        if i == m_index:
            #img_lined = cv2.line(img_org,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,255,0),2)
            img_lined= img_org
        else:
            img_lined = img_org
            #img_lined = cv2.line(img_org,(detected_lines[i][0],detected_lines[i][1]),(detected_lines[i][2],detected_lines[i][3]),(0,0,255),2)
        
    return img_lined,power_lines
    
def label_lines(lines, ransac_lines):
    left_lines=[]
    mid_lines=[]
    right_lines=[]
    other_lines=[]
    power_mast_lines =[]
    accepted_angle = np.pi/2
    l_index, m_index, r_index = find_mid_line(ransac_lines)
    for i in range(0,len(lines)):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        a,b,length = find_cart_line_eq(x1,y1,x2,y2)
        left_x_bound = int((y1-ransac_lines[l_index][1])/ransac_lines[l_index][0])-20
        right_x_bound = int((y2-ransac_lines[r_index][1])/ransac_lines[r_index][0])+20
        rho, theta, length = perpendicular_polar_line(a,b,length)
        if (x1 < left_x_bound) or (x2 < left_x_bound) or (x1 > right_x_bound) or (x2 > right_x_bound):
            other_lines.append(lines[i][0])
        else:
            
            a_l = ransac_lines[l_index][0]
            b_l = ransac_lines[l_index][1]
            theta_l = ransac_lines[l_index][3]
            close_bool = line_within_reach(a_l,b_l,x1,y1,x2,y2)
            parallel_bool = lines_approx_parallel(theta_l,theta,a_l,b_l,a,b,1920,False,accepted_angle )
            if close_bool and parallel_bool:
                left_lines.append(lines[i][0])
            else:
                a_m = ransac_lines[m_index][0]
                b_m = ransac_lines[m_index][1]
                theta_m = ransac_lines[m_index][3]
                close_bool = line_within_reach(a_m,b_m,x1,y1,x2,y2)
                parallel_bool = lines_approx_parallel(theta_m,theta,a_m,b_m,a,b,1920,False,accepted_angle )
                if close_bool and parallel_bool:
                    mid_lines.append(lines[i][0])
                    #print("Accepted as mid line ",theta_m,theta,b_m,b)
                else:
                    #print("NOT accepted as mid line ",np.rad2deg(theta_m),np.rad2deg(theta),b_m,b)
                    a_r = ransac_lines[r_index][0]
                    b_r = ransac_lines[r_index][1]
                    theta_r = ransac_lines[r_index][3]
                    close_bool = line_within_reach(a_r,b_r,x1,y1,x2,y2)
                    parallel_bool = lines_approx_parallel(theta_r,theta,a_r,b_r,a,b,1920,False,accepted_angle +0.5)
                    if close_bool and parallel_bool:
                        right_lines.append(lines[i][0])
                    else:
                        power_mast_lines.append(lines[i][0])
                    
    return left_lines,mid_lines,right_lines, other_lines , power_mast_lines

def draw_labeled_lines(lines,img,a):
    img_hough = img
    for i in range(0,len(lines)):
        #img_hough =  cv2.line(img,(lines[i][0],lines[i][1]),(lines[i][2],lines[i][3]),(a*255,255,0),2)
            #a1,b,l = find_cart_line_eq(x1,y1,x2,y2)
            #rho,theta,l = perpendicular_polar_line(a1,b,l)
            
            img_hough =  cv2.line(img,(lines[i][0],lines[i][1]),(lines[i][2],lines[i][3]),((a==0 or a == 4)*255,(a==1 or a == 3 or a ==4)*255,(a ==2 or a ==3)*255),2)#CHANGE TO 1
                
    return img_hough    

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
       
i = 0
left_x_bound = 2000
right_x_bound = 0
cap = cv2.VideoCapture(r"C:\Users\vildeg\Documents\Masteroppgave\Datasett\Discovideo\test_video.mp4")
out = cv2.VideoWriter(r"C:\Users\vildeg\Documents\Masteroppgave\Resultater\Video\"power_mast_detector.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("DONE!")
        break
    img_org=frame
    edges = cv2.Canny(img_org,350,110)
    minLineLength = 10
    maxLineGap = 8
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    binary_img = np.zeros(np.shape(img_org))
    img_lined = img_org
    #img_lined = draw_hough_lines(lines,img_org,0)
    if len(lines) == 0:
        print("No lines found.. Continues searching")
        
    else:
            
        if i == 0:
            print("Initializing")
            binary_img = draw_hough_lines(lines,binary_img,1)
            img_lined, power_lines = ThreeLineRANSAC(binary_img, img_org, 1000)
            l_index, m_index, r_index = find_mid_line(power_lines)
            
            #img_lined = cv2.line(img_lined,( int(-power_lines[l_index][1]/power_lines[l_index][0])-20,0),(int(((1080-power_lines[l_index][1])/power_lines[l_index][0])-20),1080),(0,0,0),2)
            #img_lined = cv2.line(img_lined,( int(-power_lines[r_index][1]/power_lines[r_index][0])+20,0),( int(((1080-power_lines[r_index][1])/power_lines[r_index][0])+20),1080),(0,0,0),2)
            
        elif i%50 == 0:
            print("Updating RANSAC")
            binary_img = draw_hough_lines(lines,binary_img,1)
            img_lined, power_lines = ThreeLineRANSAC(binary_img, img_org, 50)
            l_index,m_index,r_index = find_mid_line(power_lines)
            
            #img_lined = cv2.line(img_lined,( int(-power_lines[l_index][1]/power_lines[l_index][0])-20,0),(int(((1080-power_lines[l_index][1])/power_lines[l_index][0])-20),1080),(0,0,0),2)
            #img_lined = cv2.line(img_lined,( int(-power_lines[r_index][1]/power_lines[r_index][0])+20,0),( int(((1080-power_lines[r_index][1])/power_lines[r_index][0])+20),1080),(0,0,0),2)
            
        else:
            left_lines, mid_lines, right_lines, other_lines,power_mast_lines = label_lines(lines, power_lines)
            left_bool = not (len(left_lines) <= 5)
            mid_bool = not (len(mid_lines) <= 5)
            right_bool = not (len(right_lines)<= 5)
            img_lined = draw_labeled_lines(power_mast_lines,img_org,3)
            det_power_lines=[]
            #img_lined = draw_labeled_lines(other_lines,img_org,3)
            num_bins = 90
            if left_bool:
               img_lined = draw_labeled_lines(left_lines,img_org,0)
               a_l,b_l,r_l,t_l=labeled_voting_scheme(left_lines,num_bins)
               det_power_lines.append([a_l,b_l,r_l,t_l,left_bool])
            else:
               det_power_lines.append([0,0,0,0,left_bool])
            if mid_bool:
                img_lined = draw_labeled_lines(mid_lines, img_lined,1)
                a_m,b_m,r_m,t_m=labeled_voting_scheme(mid_lines,num_bins)
                det_power_lines.append([a_m,b_m,r_m,t_m,mid_bool])
            else:
               det_power_lines.append([0,0,0,0,mid_bool])
            if right_bool :
               img_lined = draw_labeled_lines(right_lines,img_lined,2)
               a_r,b_r,r_r,t_r=labeled_voting_scheme(right_lines,num_bins)
               det_power_lines.append([a_r,b_r,r_r,t_r,right_bool])
            else:
               det_power_lines.append([0,0,0,0,right_bool])
               
            need_update = not power_lines_approx_parallel(det_power_lines)#blir ikke brujkt???
            if ((mid_bool + left_bool + right_bool) <=1):# or need_update
                print("Updating RANSAC")
                binary_img = draw_hough_lines(lines,binary_img,1)
                img_lined, power_lines = ThreeLineRANSAC(binary_img, img_org, 50)
                l_index,m_index,r_index = find_mid_line(power_lines)
                for j in range(0,3):
                    x_top = -int(power_lines[j][1]/power_lines[j][0])
                    x_low = int((np.shape(img_org)[0]-power_lines[j][1])/power_lines[j][0])
                    if x_low > right_x_bound:
                        right_x_bound = x_low
                        if right_x_bound < np.shape(img_org)[1] - 20:
                            right_x_bound +=20
                        #print("yes", right_x_bound)
                    if x_top > right_x_bound:
                        right_x_bound = x_top
                        if right_x_bound < np.shape(img_org)[1] - 20:
                            right_x_bound +=20
                    if x_low < left_x_bound:
                        left_x_bound= x_low
                        if left_x_bound > 20:
                            left_x_bound -=20
                    if x_top < left_x_bound:
                        left_x_bound = x_top
                        if left_x_bound > 20:
                            left_x_bound -=20
            else:
                #print(len(power_mast_lines)," is the length of sequence")
                if len(power_mast_lines) > 0:
                    
                    detection_bool, power_mast_lines,core_line = power_mast_detector(power_mast_lines,t_l,a_l,b_l,t_m,a_m,b_m, t_r,a_r,b_r, left_bool, mid_bool, right_bool)
                    #print("Detection : ", detection_bool," Num power mast lines : ", len(power_mast_lines))
                    img_lined = draw_labeled_lines(power_mast_lines,img_org,4)
                    if detection_bool:
                        img_lined = draw_outlines_power_masts(img_lined, core_line,power_lines, l_index,r_index)
                        print("drawed outline box at (",left_x_bound,",",core_line[1],")")
                #else if right_bool:
                            
            
                        #detection_bool, core_line = power_mast_detector(power_mast_lines,t_m)
                
                #img_lined = cv2.line(img_lined,(left_x_bound,0),(left_x_bound,2000),(0,0,0),2)
                #img_lined = cv2.line(img_lined,(right_x_bound,0),(right_x_bound,2000),(0,0,0),2)
            #else:
                
             #  img_lined = draw_result_line(a_r,b_r,img_lined, 0,0,255)
              # img_lined = draw_result_line(a_l,b_l,img_lined, 255,0,0)
               #img_lined = draw_result_line(a_m,b_m,img_lined, 0,255,0)
                    
            #if len(other_lines)!= 0:
               # img_lined = draw_labeled_lines(other_lines, img_lined,3)
                
    cv2.imshow('frame',img_lined)
    out.write(img_lined)
    i += 1
    k = cv2.waitKey(1)
    if k == 27:         # wait for ESC key to exit
        cap.release()
        #out.release()
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        #save file heres
        cap.release()
        #sout.release()
        cv2.destroyAllWindows()
