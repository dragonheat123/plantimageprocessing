import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import math
import pandas as pd




os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel1_vert')

#dirlist = glob.glob('top_*.jpg')
dirlist = glob.glob('img*.jpg')
dirlist.sort(key=os.path.getmtime)


for i in range(0,len(dirlist)-1): 
    g1 = cv2.imread(dirlist[i])
    g2 = cv2.imread(dirlist[i+1])      
    f = cv2.absdiff(g1, g2)
    
    r = f[:,:,2].copy()
    b = f[:,:,0].copy()
    cv2.imshow('g',g2)
    cv2.imshow('f',f)
    cv2.imshow('r',r)
    cv2.imshow('b',b)
    cv2.waitKey(0)