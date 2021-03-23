# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 17:55:14 2021

@author: srika
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import glob

os.chdir(r'D:\Projects\OpenCV\Bottles')
cwd=os.getcwd()

count=0
for names in glob.glob('D:\Projects\OpenCV\Bottles\[0-9].*'):
    img=cv2.imread(names)
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(imgray,127,255,0)
    contours,hirerarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    newcontour=[]
    newcontour.append(contours[0])

    max=0;
    for i in range(0,len(contours)): 
        if(contours[i].shape[0]>max):
            newcontour[0]=contours[i]  
            max=contours[i].shape[0]

    img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.drawContours(img1,newcontour,-1,(255,0,0),4)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(img1)
    plt.savefig('Bottle'+str(count)+'png',dpi=300)
    count=count+1
    plt.figure()