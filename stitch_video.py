import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import math

startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())
datearray = list(range(startdate,enddate,60*60))

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel1')
panel1_h = glob.glob('*.jpg')
panel1_h.sort(key=os.path.getmtime)
panel1_h = ['C://Users//xxx//Desktop//smartfarm_data//panel1//' + s for s in panel1_h]

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel2')
panel2_h = glob.glob('*.jpg')
panel2_h.sort(key=os.path.getmtime)
panel2_h = ['C://Users//xxx//Desktop//smartfarm_data//panel2//' + s for s in panel2_h]

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel3')
panel3_h = glob.glob('*.jpg')
panel3_h.sort(key=os.path.getmtime)
panel3_h = ['C://Users//xxx//Desktop//smartfarm_data//panel3//' + s for s in panel3_h]

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel1_vert')
panel1_v = glob.glob('*.jpg')
panel1_v.sort(key=os.path.getmtime)
panel1_v = ['C://Users//xxx//Desktop//smartfarm_data//panel1_vert//' + s for s in panel1_v]

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel2_vert')
panel2_v = glob.glob('*.jpg')
panel2_v.sort(key=os.path.getmtime)
panel2_v = ['C://Users//xxx//Desktop//smartfarm_data//panel2_vert//' + s for s in panel2_v]

os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel3_vert')
panel3_v = glob.glob('*.jpg')
panel3_v.sort(key=os.path.getmtime)
panel3_v = ['C://Users//xxx//Desktop//smartfarm_data//panel3_vert//' + s for s in panel3_v]

for k in range(0,len(datearray)):
    stitch = np.zeros((480*2,640*3,3)).astype('uint8')
    stitch[0:480,0:640,0:3] = cv2.imread(panel1_h[k])
    stitch[480:480*2,0:640,0:3] = cv2.imread(panel1_v[k])
    stitch[0:480,640:640*2,0:3] = cv2.imread(panel2_h[k])
    stitch[480:480*2,640:640*2,0:3] = cv2.imread(panel2_v[k])
    stitch[0:480,640*2:640*3,0:3] = cv2.imread(panel3_h[k])
    stitch[480:480*2,640*2:640*3,0:3] = cv2.imread(panel3_v[k])
    cv2.imwrite("C://Users//xxx//Desktop//smartfarm_data//stitch//img-"+str(k)+".jpg",stitch)
    print(k)
    
    
os.chdir("C://Users//xxx//Desktop//smartfarm_data//stitch//")
os.system('ffmpeg -framerate 15 -i img-%d.jpg -c:v libx264 -r 60 -pix_fmt yuv420p out.mp4')
    
    
    