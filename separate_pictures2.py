import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

os.chdir('C://Users//xxx//Desktop//Dropbox//pics (2)')
dirlist = glob.glob('top2_*.jpg')
dirlist = dirlist[0:len(dirlist)-1]

area_array = []
percentage_array = []

for i in dirlist:
    if int(i[5:15])>1533119403 and int(i[5:15])<1534847403:
        img = cv2.imread(i)
        
        datenum = i[5:15]
        #cv2.imshow('now',img)
        #
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
        #
        #
        #pts1 = np.array([[54,114],[501,82],[59,448],[550,385]])
        pts1 = np.array([[50,65],[496,79],[53,407],[505,375]])
        pts2 = np.array([[0,0],[640,0],[0,480],[640,480]])
        
        h, status = cv2.findHomography(pts1, pts2)												### correcing images
        #
        dst = cv2.warpPerspective(img_undistorted,h,(640,480))
        
        #cv2.imshow('dst',dst)
        #
        res2 = dst.copy()
        
        cv2.imwrite('C://Users//xxx//Desktop//smartfarm_data//panel2//'+i[5:15]+'.jpg',res2)
        
        
        print(i[5:15])
        
        r = res2[:,:,2].copy()
        b = res2[:,:,0].copy()
        
        ### detect black pixels in cropped image
        small = r[230:370,250:390].copy()
        #cv2.imshow('cropped',small)
        ret2,small2 = cv2.threshold(small,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('small2',small2)
        percentblack = (140*140-cv2.countNonZero(small2))/(140*140)*100
        print('percentage black:',percentblack,'%')
        
        
        
        #cv2.imshow('r0',r)
        ##adding brightness on the right
        for i in range(0,140):
            r[0:480,500+i] += min(round(50/140*i),255)
            b[0:480,500+i] += min(round(30/140*i),255)
        
        #cv2.imshow('r1',r)
        #cv2.imshow('b1',b)
        
        #print(np.mean(r),np.mean(b))
        
        
        #####include something adaptive thresholding here
        #####
        threshold_val = percentblack
        
        r[r<=np.percentile(r,threshold_val)]=0
        r[r>np.percentile(r,threshold_val)]=255
        #cv2.imshow('r',r)
        #
        b[b<=np.percentile(b,threshold_val)]=0
        b[b>np.percentile(b,threshold_val)]=255
        
        #cv2.imshow('b',b)
        
        r = cv2.bitwise_not(r)
        b= cv2.bitwise_not(b)
        
        ###otsu thresholding
        #ret2,r = cv2.threshold(r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret2,b = cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #r = cv2.bitwise_not(r)
        #b = cv2.bitwise_not(b)
        
        th = cv2.bitwise_not(cv2.bitwise_and(r,b))
        #cv2.imshow('th',th)
        #th = r*th+b*th
        #th = (255*th).astype('uint8')
        
        #
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        
        #cv2.imshow('open',opening)
        
        opening = 255 - opening
        
        opening[0:480,0] = 0
        opening[0:480,639] = 0
        opening[0,0:639] = 0
        opening[479,0:639] = 0
        alpha = 0.3
        im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)			### drawing out contour of leaves
        contours2 = []
        for i in contours:
            if cv2.contourArea(i)>10:
                contours2.append(i)
        
        draw = cv2.drawContours(dst, contours2, -1, (0,255,0), 1)
        overlay = draw.copy()
        output = draw.copy()
        cv2.fillPoly(overlay, pts =contours2, color=(0,255,255))
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        #cv2.imshow('drawn',output)
        
        area = 0
        
        for i in contours:
            if cv2.contourArea(i)>10:
                area += cv2.contourArea(i)
        
        area = area/(49)/20           #35px = 5cm   for 20 plants
            
        print('avgleafarea:' , area)
        
        area_array.append(area)
        percentage_array.append(percentblack)
        
        cv2.imwrite('C://Users//xxx//Desktop//smartfarm_data//panel2_seg//area_'+datenum+'.jpg',output)
        
plt.plot(area_array)
plt.show()

plt.plot(percentage_array)
plt.show()