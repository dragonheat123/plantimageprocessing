import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.cm as cm
import datetime
import math

#os.chdir('C://Users//xxx//Desktop//smartfarm_data//panel1')
#dirlist = glob.glob('*.jpg')
#dirlist.sort(key=os.path.getmtime)
import time
tic = time.time()

#mat = [0.306,0.335,0.367,0.402,0.4,0.38,0.529,0.51,0.51,0.58,0.71,0.76,0.915,0.95,0.95,0.95,0.95,0.95,0.95,0.95] #panel1
#mat = [0.306,0.335,0.367,0.402,0.4,0.38,0.529,0.51,0.51,0.515,0.65,0.67,0.75,0.78,0.95,0.95,0.95,0.95,0.95,0.95] #panel 2
mat = [0.306,0.335,0.367,0.402,0.4,0.41,0.529,0.51,0.54,0.58,0.71,0.76,0.915,0.95,0.95,0.95,0.95,0.95,0.95,0.95] #panel3


#os.chdir('D://Research Data//pics (1)//')
os.chdir('C://Users//xxx//Desktop//Dropbox//pics (2)')
dirlist = glob.glob('top3_*.jpg')
dirlist = dirlist[0:len(dirlist)-1]
#dirlist = [i for i in dirlist if np.mean(cv2.imread(i))>65]
dirlist_time = []

arr=[]
arrpercent = []

##1st bath top 3july to 23 july
##1st batch top2 19jun to 8 jul
##1st batch top3 
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
    
tic = time.time()    
#center1
#centers = [(33,71),(32,193),(27,311),(27,438),(30,562),(163,63),(167,181),(165,310),(167,435),(168,562),(304,65),(305,190),(308,315),(305,440),(304,565),(448,69),(448,190),(448,314),(447,446),(446,565)]
#center2
#centers =  [(25,72),(26,198),(26,324),(26,451),(26,568),(168,65),(168,196),(168,330),(168,452),(168,573),(304,66),(305,196),(310,323),(313,448),(305,565),(441,59),(442,193),(446,319),(447,436),(447,556)]
#center3
centers = [(26,62),(26,185),(16,310),(15,443),(19,575),(158,55),(154,185),(154,316),(154,451),(154,573),(302,61),(301,190),(296,326),(294,462),(294,586),(446,64),(445,193),(440,335),(442,461),(435,589)]

centers = [(k[0]/2,k[1]/2) for k in centers]
distcenter = np.zeros([240,320])
for i in range(0,240):
    for j in range(0,320):
        distcenter[i,j] = np.min([np.linalg.norm(np.array((i,j))-np.array(x)) for x in centers])
    
imgdex = 0    

arr = []
for k in datearray[0:480:24]:
    imgdex+=1
    print(imgdex)
    img1 = cv2.imread(dirlist[find_nearest(dirlist_time,k)])
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
    img_undistorted = cv2.fisheye.undistortImage(img1, K, D=D, Knew=Knew)
    #cv2.imwrite('fisheye_sample_undistorted.jpg', img_undistorted)
    #cv2.imshow('undistorted', img_undistorted)
    

    #pts1 = np.array([[136,86],[591,65],[141,377],[576,402]])                              #for top1
    #pts1 = np.array([[60,75],[495,85],[68,410],[510,375]])                                #for top2
    pts1 = np.array([[80,118],[480,95],[90,439],[515,390]])                               #for top3
    pts2 = np.array([[0,0],[640,0],[0,480],[640,480]])
    
    h, status = cv2.findHomography(pts1, pts2)												### correcing images
    #
    img2 = cv2.warpPerspective(img_undistorted,h,(640,480))
    img = img2.copy()
    
    print('time for dist:',time.time()-tic)
    
    #img = cv2.imread('C://Users//xxx//Desktop//Dropbox//samplepics//top_now.jpg')		###load images
    #imgn = 0
    
    #img = cv2.imread(dirlist[imgn])
    #plt.imshow(img)
    labimg= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = labimg[:,:,0]
#    cv2.imshow('l',L)
    laplacian = cv2.Laplacian(L,cv2.CV_64F).astype('float64')
    laplacian = (laplacian-np.min(laplacian))/(np.max(laplacian)-np.min(laplacian))
    
    plt.figure(1)
    #plt.hist(L.ravel(),256,[0,256]);        
#    cv2.imshow('lab',L)
    
    for i in range(0,160):
        for j in range(0,480):
            L[j,640-160+i] = min(L[j,640-160+i]+int(50/160*i),255)
            
    for i in range(0,100):
        for j in range(0,480):
            L[j,i] = min(L[j,i]+40-int(40/100*i),255)        
    
    import imutils
    L = imutils.resize(L,width=320)
    
    
            
    distcenter_std = (distcenter-np.min(distcenter))/(np.max(distcenter)-np.min(distcenter))
    
    ###
    
    L_std = L.astype(float)/255
    
    array = np.zeros([L.shape[0]*L.shape[1],2])
    
    L_std_f = L_std.flatten()
    #L_std_f = np.round(L_std_f*8)/8
    distcenter_std_f = distcenter_std.flatten()
    #distcenter_std_f = np.round(distcenter_std_f*8)/8
    laplacian_std_f = laplacian.flatten()
    
    array[:,0] = L_std_f
    array[:,1] = distcenter_std_f
    
    
    from sklearn.cluster import KMeans
    
  #  from sklearn.cluster import MiniBatchKMeans
    
    labels = []
    costs = []                              ###dist from datapoints in cluster to centroid
    data = list(zip(L_std_f,distcenter_std_f,laplacian_std_f))
    clus = 9
    #kmeans_model = MiniBatchKMeans(n_clusters=9, batch_size=1000).fit(data)
    kmeans_model = KMeans(n_clusters=clus, precompute_distances=True).fit(data)
    labels.append(kmeans_model.labels_)
    costs.append(kmeans_model.inertia_)
    dist = []                                           ###elbow method to decide number of ppl
    #for i in range(0,len(costs)):
    #    a = np.linalg.norm(np.array([10,costs[len(costs)-1]])-np.array([2,costs[0]]))
    #    b = np.linalg.norm(np.array([i,costs[i]])-np.array([2,costs[0]]))
    #    c=  np.linalg.norm(np.array([10,costs[len(costs)-1]])-np.array([i,costs[i]]))      
    #    s= 0.5*(a+b+c)
    #    area = (s*(s-a)*(s-b)*(s-c))**0.5
    #    dist.append(2*area/a)
    #print('best K: ',dist.index(max(dist))+2)
    #plt.figure(3)
    #plt.plot(costs)
    
    
    import matplotlib
    plt.figure(10)
    #plt.scatter(L_std_f,distcenter_std_f,  c=[matplotlib.cm.get_cmap('Paired')(i) for i in labels[0]], cmap=plt.get_cmap('Accent'),label='r');
    for i in range(0,clus):
        plt.scatter(L_std_f[labels[0]==i],distcenter_std_f[labels[0]==i],  c=[matplotlib.cm.get_cmap('Paired')(j) for j in labels[0][labels[0]==i]], cmap=plt.get_cmap('Accent'),label=str(i));
    plt.xlabel('Brightness of Color')
    plt.ylabel('Distance from Center')
    plt.legend()
    plt.savefig("C://Users//xxx//Desktop//smartfarm_data//panel3_kseg//scat-"+str(imgdex)+".jpg")
    plt.ioff()
    
    
    label_rs = np.array([matplotlib.cm.get_cmap('Paired')(i) for i in labels[0]]).reshape(L.shape[0],L.shape[1],4)
    plt.figure(11)
    plt.imshow(label_rs)
    plt.savefig("C://Users//xxx//Desktop//smartfarm_data//panel3_kseg//seg-"+str(imgdex)+".jpg")
    plt.ioff()
    
    
    #plt.figure(12)
    #plt.imshow(L)
    
    #
    dist = []
    light =[]
    norm = []
    print('i', 'light1','dist1')
    for i in range(0,max(labels[0]+1)):
        dist1 = np.mean(distcenter_std_f[labels[0] == i])
        light1 = np.mean(L_std_f[labels[0] == i])
        dist.append(dist1)
        light.append(light1)
        norm.append(np.sqrt(dist1**2+light1**2))
        print(i, light1,dist1,np.sqrt(dist1**2+light1**2))
    
    print('time:', time.time()-tic)
    norm = [norm.index(k) for k in norm if k < mat[imgdex-1]]
    print('thres:', mat[imgdex-1])
    #
    
    ####contours
    img = img2  #1708/1942
    
    seg = np.zeros([L.shape[0],L.shape[1]])
    label1 = labels[0].reshape(L.shape[0],L.shape[1]) 
    for i in norm:
        seg[label1==i] = 255
    seg = imutils.resize(seg,width=img.shape[1])
    #cv2.imshow('seg',seg)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('open',opening)
    
    opening = (opening).astype('uint8')
    
    opening[0:480,0] = 0
    opening[0:480,639] = 0
    opening[0,0:639] = 0
    opening[479,0:640] = 0
    alpha = 0.3
    im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)			### drawing out contour of leaves
    contours2 = []
    for i in contours:
        if cv2.contourArea(i)>10:
            contours2.append(i)
            
    draw = cv2.drawContours(img.copy(), contours2, -1, (0,255,0), 1)
    overlay = draw.copy()
    output = draw.copy()
    cv2.fillPoly(overlay, pts =contours2, color=(0,255,255))
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
#    cv2.imshow('output',output)
    
    area = 0
    
    for i in contours:
        if cv2.contourArea(i)>10:
            area += cv2.contourArea(i)
    
    area = area/(49)/16   
    print('area: ', area)
    
    cv2.putText(output,datetime.datetime.fromtimestamp(k).strftime("%A %d %B %Y %I:%M:%S%p"),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.putText(output,"est. area per plant: "+str(round(area,2))+" cm2",(10,output.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
    cv2.imwrite("C://Users//xxx//Desktop//smartfarm_data//panel3_kseg//img-"+str(imgdex)+".jpg",output)
    plt.close('all')
    
    
    arr.append(area)
    

import pandas as pd
df = pd.DataFrame(arr, columns=["panel3areaK"])
df.to_csv('C://Users//xxx//Desktop//smartfarm_data//panel3K.csv', index=False)

plt.ion()
plt.figure(1010)
plt.plot(list(range(0,20)),arr)

img = cv2.imread('C://Users//xxx//Desktop//smartfarm_data//panel3_kseg//img-5.jpg')

plt.imshow('C://Users//xxx//Desktop//smartfarm_data//panel3_kseg//img-5.jpg')
    