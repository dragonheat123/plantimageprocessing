import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import math

##### plotting color spaces
#    grid.imshow(img[:,:,2],cmap=cm.gray)
#    grid.set_title('R')
#    
#    labimg= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#    grid = plt.subplot(434)
#    grid.imshow(labimg[:,:,0],cmap=cm.gray)
#    grid.set_title('L')
#    grid = plt.subplot(435)
#    grid.imshow(labimg[:,:,1],cmap=cm.gray)
#    grid.set_title('A')
#    grid = plt.subplot(436)
#    grid.imshow(labimg[:,:,2],cmap=cm.gray)
#    grid.set_title('B')
#    
#    ycb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#    grid = plt.subplot(437)
#    grid.imshow(ycb[:,:,0],cmap=cm.gray)
#    grid.set_title('Y')
#    grid = plt.subplot(438)
#    grid.imshow(ycb[:,:,1],cmap=cm.gray)
#    grid.set_title('C')
#    grid = plt.subplot(439)
#    grid.imshow(ycb[:,:,2],cmap=cm.gray)
#    grid.set_title('B')
#    
#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    grid = fig.add_subplot(4,3,10)
#    grid.imshow(hsv[:,:,0],cmap=cm.gray)
#    grid.set_title('H')
#    grid = fig.add_subplot(4,3,11)
#    grid.imshow(hsv[:,:,1],cmap=cm.gray)
#    grid.set_title('S')
#    grid = fig.add_subplot(4,3,12)
#    grid.imshow(hsv[:,:,2],cmap=cm.gray)
#    grid.set_title('V')
#    fig.savefig("C://Users//xxx//Desktop//smartfarm_data//colorspaces//"+dirlist[i])
#    plt.ioff()


os.chdir('D://Research Data//pics (1)//')

dirlist = glob.glob('top_*.jpg')
#dirlist = glob.glob('top2_*.jpg')


dirlist = dirlist[0:len(dirlist)-1]
dirlist = [i for i in dirlist if np.mean(cv2.imread(i))>65]
dirlist_time = []

arr=[]
arrpercent = []

##1st bath top 3july to 23 july
##1st batch top2 19jun to 8 jul
##1st batch top3 
startdate = round(datetime.datetime(2018,7,3,0,0).timestamp())
enddate = round(datetime.datetime(2018,7,22,0,0).timestamp())

for i in dirlist:   
    dirlist_time.append(int(i[4:14]))
    
    
datearray = list(range(startdate,enddate,60*60))

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
    K = np.array([[  305,     0.  ,  320],													### camera dependent focus length and image centre settings
                  [    0.  ,   305,   240],
                  [    0.  ,     0.  ,     1.  ]])
    #
    # zero distortion coefficients work well for this image
    D = np.array([0., 0., 0., 0.])
    #
    ## use Knew to scale the output
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.6 * Knew[(0,1), (0,1)]
    #
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    #cv2.imwrite('fisheye_sample_undistorted.jpg', img_undistorted)
    #cv2.imshow('undistorted', img_undistorted)
    

    pts1 = np.array([[136,86],[591,65],[141,377],[576,402]])                              #for top1
    #pts1 = np.array([[60,75],[495,85],[68,410],[510,375]])                                #for top2
    #pts1 = np.array([[80,118],[480,95],[90,439],[515,390]])                               #for top3
    pts2 = np.array([[0,0],[640,0],[0,480],[640,480]])
    
    h, status = cv2.findHomography(pts1, pts2)												### correcing images
    #
    dst = cv2.warpPerspective(img_undistorted,h,(640,480))
    
    #cv2.imshow('dst',dst)
    #cv2.imwrite('undistorted.jpg',dst)
    #cv2.imshow('dst',dst)
    #cv2.waitKey()
    #
    res2 = dst.copy()

    r = res2[:,:,2].copy()
    b = res2[:,:,0].copy()
   
    
    crop_img1=r[120:360,128:384]

    ret2,th2 = cv2.threshold(crop_img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    numBlackPixel=np.sum(th2[:]<=ret2)
    
    numTot=th2.size
    
    if imgdex<=250:
        perc=(np.float(numBlackPixel)/numTot)*100
        
    else:
        perc= 50 + (90-50)/(480-250)*(imgdex-250)
    
    arrpercent.append(perc)
    threshold_val= perc
    
    r[r<=np.percentile(r,threshold_val)]=0
    r[r>np.percentile(r,threshold_val)]=255
    
    b[b<=np.percentile(b,threshold_val)]=0
    b[b>np.percentile(b,threshold_val)]=255
    
    r = cv2.bitwise_not(r)
    b= cv2.bitwise_not(b)
    
    th = cv2.bitwise_not(cv2.bitwise_and(r,b))

    th=th[:,0:500]
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    
    
    opening = 255 - opening
    
    opening[0:480,0] = 0
    opening[0:480,499] = 0
    opening[0,0:499] = 0
    opening[479,0:499] = 0
    alpha = 0.3
    im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)			### drawing out contour of leaves
    contours2 = []
    for i in contours:
        if cv2.contourArea(i)>10:
            contours2.append(i)
    
    #draw = cv2.drawContours(dst[:,0:500], contours2, -1, (0,255,0), 1)
    draw = cv2.drawContours(dst, contours2, -1, (0,255,0), 1)
    overlay = draw.copy()
    output = draw.copy()
    cv2.fillPoly(overlay, pts =contours2, color=(0,255,255))
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.imshow('output',output)


    #path = r"C:\Users\SUTD\Desktop\top1_cont"
    #cv2.imwrite(os.path.join(path ,name), output)
    area = 0
    
    for i in contours:
        if cv2.contourArea(i)>10:
            area += cv2.contourArea(i)
    
    area = area/(49)/16          #35px = 5cm   for 16 plants
        
    arr.append(round(area,2))
    cv2.putText(res2,datetime.datetime.fromtimestamp(k).strftime("%A %d %B %Y %I:%M:%S%p"),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.putText(res2,"est. area per plant: "+str(round(area,2))+" cm2",(10,dst.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    #cv2.imshow('dst',res2)
    cv2.imwrite("C://Users//xxx//Desktop//smartfarm_data//panel1_1st//img-"+str(imgdex)+".jpg",res2)
    cv2.putText(output,datetime.datetime.fromtimestamp(k).strftime("%A %d %B %Y %I:%M:%S%p"),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.putText(output,"est. area per plant: "+str(round(area,2))+" cm2",(10,dst.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.imwrite("C://Users//xxx//Desktop//smartfarm_data//panel1_1st_seg//img-"+str(imgdex)+".jpg",output)
    
#os.chdir("D://Research Data//panel1_1st//")
#os.system('ffmpeg -framerate 15 -i img-%d.jpg -c:v libx264 -r 60 -pix_fmt yuv420p out.mp4')
#os.chdir("D://Research Data//panel2_1st_seg//")
#os.system('ffmpeg -framerate 15 -i img-%d.jpg -c:v libx264 -r 60 -pix_fmt yuv420p out.mp4')

import pandas as pd
df = pd.DataFrame(arr, columns=["panel1area_1st"])
df.to_csv('C://Users//xxx//Desktop//smartfarm_data//panel1_1h_1st.csv', index=False)