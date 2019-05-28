# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:00:32 2019

@author: vildeg
"""
import cv2
import numpy as np
import time

from RANSAC_algorithm import RANSAC_alg 
from RANSAC_algorithm import draw_RANSAC_results 

from AMA_RANSAC_algorithm import AMA_RANSAC_alg 
from AMA_RANSAC_algorithm import draw_AMA_RANSAC_results 

from HTIVS_algorithm import HTIVS_alg
from HTIVS_algorithm import draw_HTIVS_results

from Kmeans_algorithm import KMeans_alg
from Kmeans_algorithm import draw_KMEANS_results

from AMA_RANSAC_HTIVS_algorithm import AMA_RANSAC_HTIVS_alg
from AMA_RANSAC_HTIVS_algorithm import draw_AMA_RANSAC_HTIVS_results

from AMA_RANSAC_LABEL_algorithm import AMA_RANSAC_LABEL_alg 
from AMA_RANSAC_LABEL_algorithm import draw_AMA_RANSAC_LABEL_results 


alg_list = [0,0,0,1,1] # RANSAC, AMA RANSAC, HTIVS,  K-MEANS, AMA RANSAC + HTIVS
acc_list=np.zeros(2) # [number of good result,number of bad result]
time_list = np.zeros(3) # [best time, worst time, avg time ]
time_list[0]=10000
i = 0
show = True
cap = cv2.VideoCapture(r"C:\Users\vildeg\Documents\Masteroppgave\Datasett\Discovideo\Evaluation_set\easy_w_horizon.mp4")
y = 0
while(cap.isOpened()):
    
    ret, frame = cap.read()
    if not ret:
        print("DONE!")
        print("RANSAC rerun ",y," times")
        print("Accuracy = ", acc_list[0]/(i) ," = ", acc_list[0], "/" , i)
        print("Best time = " , time_list[0])
        print("Worst time = ", time_list[1])
        print("Avg time = ", time_list[2])
        print("ransac rerunned ", y, "times")
        break
    i += 1
    show = True
    #HER SKAL DET INN
    if alg_list[0]==1:#RANSAC
        start = time.time()
        detected_lines = RANSAC_alg(frame)
        end = time.time()
        img_result = draw_RANSAC_results(detected_lines,frame)
    elif alg_list[1]==1:#AMA_RANSAC
        start = time.time()
        detected_lines = AMA_RANSAC_alg(frame)
        end = time.time()
        img_result = draw_AMA_RANSAC_results(detected_lines,frame)
    elif alg_list[2] == 1 : #HTIVS
        start = time.time()
        a,b,rho,theta= HTIVS_alg(frame)
        end = time.time()
        img_result = draw_HTIVS_results(a,b,frame)
    elif alg_list[3] == 1: #Kmeans + HTIVS (RANSAC)
        if i == 0:
            detected_lines = RANSAC_alg(frame)
            show = False
        else:
            start = time.time()
            a,b,update_needed = KMeans_alg(frame, detected_lines)
            if update_needed:
                print("ransac runned")
                y += 0
                detected_lines = AMA_RANSAC_alg(frame)
                end = time.time()
                img_result = draw_AMA_RANSAC_results(detected_lines,frame)
            else:
                end = time.time()
                img_result = draw_KMEANS_results(a,b,frame)
    
    elif alg_list[4] == 1: # AMA RANSAC + HTIVS
        
        if i == 0:
            detected_lines = RANSAC_alg(frame)
            show = False
        else:
           # print("her first")
            start = time.time()
            power_lines,update_needed = AMA_RANSAC_HTIVS_alg(frame, detected_lines)
            if update_needed or (i % 10 == 0):
                y += 1
                #print("her alltid ? ",y)
                detected_lines = AMA_RANSAC_alg(frame)
                end = time.time()
                img_result = draw_AMA_RANSAC_results(detected_lines,frame)
            else:
                end = time.time()
                img_result = draw_AMA_RANSAC_HTIVS_results(frame, power_lines)
   # elif alg_list[5] == 1: # AMA RANSAC + LABELS
    #    start = time.time()
     #   detected_lines = AMA_RANSAC_LABEL_alg(frame)
      #  end = time.time()
       # img_result = draw_AMA_RANSAC_LABEL_results(detected_lines,frame)
                
                
        
    
    
    
    
    #print("Frame ",i)
    
    if show == True:
        cv2.imshow('frame',img_result)
        k = cv2.waitKey(1)
        
        t = end-start
        if t < time_list[0] or i == 1:
            time_list[0] = t
        if t > time_list[1]:
            time_list[1] = t
        time_list[2] = (time_list[2]*(i-1)+t)/i
        if k == ord('s'): # wait for 's' key to save and exit
            #save file heres
            print("RANSAC rerun ",y," times")
            print("Accuracy = ", acc_list[0]/(i) ," = ", acc_list[0], "/" , i)
            print("Best time = " , time_list[0])
            print("Worst time = ", time_list[1])
            print("Avg time = ", time_list[2])
            cap.release()
            #sout.release()
            cv2.destroyAllWindows()
        elif k == ord('1'):
            acc_list[0]+=1
        elif k == ord('2'):
            acc_list[1]+=1