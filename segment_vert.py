# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 15:19:37 2018

@author: xxx
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import math

os.chdir('C://Users//xxx//Desktop//Dropbox//pics (2)')
dirlist = glob.glob('vert2_*.jpg')

dirlist = dirlist[0:len(dirlist)-1]
dirlist_time = []

arr=[]
arrpercent = []

startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

for i in dirlist:   
    dirlist_time.append(int(i[5:15]))
    
datearray = list(range(startdate,enddate,60*60*24))

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx
    else:
        return idx
    
imgdex = 0    

for k in datearray:
    imgdex+=1
    print(imgdex)
    img = cv2.imread(dirlist[find_nearest(dirlist_time,k)])        
    cv2.putText(img,datetime.datetime.fromtimestamp(k).strftime("%A %d %B %Y %I:%M:%S%p"),(10,img.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.imshow('img',img)
    cv2.imwrite("C://Users//xxx//Desktop//smartfarm_data//panel1_vseg//img-"+str(imgdex)+".jpg",img)
    
os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel1_vseg')

dirlist = 'C://Users//xxx//Desktop//smartfarm_data//panel1_vseg//img-20.jpg'

img = cv2.imread(dirlist)

plt.imshow(img)

h = 107

cv2.rectangle(img,(0,h),(639,h),(0,255,0),1)
cv2.putText(img,"leaf height: "+str(480-h)+'px',(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)

cv2.imshow('img',img)

cv2.imwrite(dirlist,img)
    

arr = [75,88,94,106,109,109,105,124,146,150,158,145,156,173,211,238,259,320,347,373]

import pandas as pd
df = pd.DataFrame(arr, columns=["panel1_vert"])
df.to_csv('C://Users//xxx//Desktop//smartfarm_data//panel1_vert.csv', index=False)

plt.plot(df)

    
    