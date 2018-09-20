import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import math

os.chdir('C://Users//xxx//Desktop//Dropbox//pics (2)')
#dirlist = glob.glob('top_*.jpg')
#dirlist = glob.glob('top2_*.jpg')
dirlist = glob.glob('top3_*.jpg')

dirlist = dirlist[0:len(dirlist)-1]
dirlist_time = []

arr=[]
arrpercent = []

startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

for i in dirlist:   
    dirlist_time.append(int(i[5:15]))
    
datearray = list(range(startdate,enddate,60*60))

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx
    else:
        return idx
    
imgdex = 0    

for k in datearray[1:301]:
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
    

    #pts1 = np.array([[136,86],[591,65],[141,377],[576,402]])                              #for top1
    #pts1 = np.array([[60,75],[495,85],[68,410],[510,375]])                                #for top2
    pts1 = np.array([[80,118],[480,95],[90,439],[515,390]])                               #for top3
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

#    cv2.imshow('rb',r)
#    cv2.imshow('bb',b)
#    
#    for i in range(0,160):
#        for j in range(0,480):
#            r[j,640-160+i] = min(r[j,640-160+i]+int(50/160*i),255)
#            b[j,640-160+i] = min(b[j,640-160+i]+int(30/160*i),255)
#        
#    for i in range(0,100):
#        for j in range(0,480):
#            r[j,i] = min(r[j,i]+40-int(40/100*i),255)
#            b[j,i] = min(b[j,i]+20-int(20/100*i),255)  
#    
#    cv2.imshow('r2',r)
#    cv2.imshow('b2',b)             
    
    
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
    cv2.imshow('r',r)
    cv2.imshow('b',b)
    
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
    cv2.imshow('dst',res2)
    #cv2.imwrite("C://Users//xxx//Desktop//smartfarm_data//panel3//img-"+str(imgdex)+".jpg",res2)
    cv2.putText(output,datetime.datetime.fromtimestamp(k).strftime("%A %d %B %Y %I:%M:%S%p"),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.putText(output,"est. area per plant: "+str(round(area,2))+" cm2",(10,dst.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    #cv2.imwrite("C://Users//xxx//Desktop//smartfarm_data//panel3_seg//img-"+str(imgdex)+".jpg",output)


#plt.plot(arr)  
#plt.plot(arrpercent)  
#plt.show()

#import pandas as pd
#df = pd.DataFrame(arr, columns=["panel3area"])
#df.to_csv('C://Users//xxx//Desktop//smartfarm_data//panel3.csv', index=False)

#os.chdir("C://Users//xxx//Desktop//smartfarm_data//panel3//")
#os.system('ffmpeg -framerate 15 -i img-%d.jpg -c:v libx264 -r 60 -pix_fmt yuv420p out.mp4')
#os.chdir("C://Users//xxx//Desktop//smartfarm_data//panel3_seg//")
#os.system('ffmpeg -framerate 15 -i img-%d.jpg -c:v libx264 -r 60 -pix_fmt yuv420p out.mp4')
