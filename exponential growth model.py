import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import math
import pandas as pd
import random
random.seed(2)

startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

datearray = list(range(startdate,enddate,60*60))

readdate = []
#
#for i in datearray:
#    #readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m/%y %H:%M"))
#    readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m"))


panel1area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel1k.csv', delimiter=',')
panel2area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel2k.csv', delimiter=',')
panel3area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel3k.csv', delimiter=',')
alldata = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//allpanel_d.csv',delimiter=',')
alldata1 = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//allpanel_hr_19jun_23jul.csv',delimiter=',')
panel1area1h = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel.csv', delimiter=',')
alldata1h = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//allpanel_hr_2aug_2sep.csv', delimiter=',')

alldata = alldata.bfill()

panel1avgred = alldata.ix[:,['ppfdred1','ppfdred2','ppfdred3','ppfdred4','ppfdred5','ppfdred6','ppfdred7']]
panel1avgblue = alldata.ix[:,['ppfdblue1','ppfdblue2','ppfdblue3','ppfdblue4','ppfdblue5','ppfdblue6','ppfdblue7']]
panel2avgred = alldata.ix[:,['ppfdred1_2','ppfdred2_2','ppfdred3_2','ppfdred4_2','ppfdred5_2','ppfdred6_2','ppfdred7_2']]
panel2avgblue = alldata.ix[:,['ppfdblue1_2','ppfdblue2_2','ppfdblue3_2','ppfdblue4_2','ppfdblue5_2','ppfdblue6_2','ppfdblue7_2']]
panel3avgred = alldata.ix[:,['ppfdred1_3','ppfdred2_3','ppfdred3_3','ppfdred4_3','ppfdred5_3','ppfdred6_3','ppfdred7_3']]
panel3avgblue = alldata.ix[:,['ppfdblue1_3','ppfdblue2_3','ppfdblue3_3','ppfdblue4_3','ppfdblue5_3','ppfdblue6_3','ppfdblue7_3']]
panel1avgred1h = alldata1h.ix[:,['ppfdred1','ppfdred2','ppfdred3','ppfdred4','ppfdred5','ppfdred6','ppfdred7']]
panel1avgblue1h = alldata1h.ix[:,['ppfdblue1','ppfdblue2','ppfdblue3','ppfdblue4','ppfdblue5','ppfdblue6','ppfdblue7']]

panel1avgred = panel1avgred.mean(axis=1)
panel1avgblue = panel1avgblue.mean(axis=1)
panel2avgred = panel2avgred.mean(axis=1)
panel2avgblue = panel3avgblue.mean(axis=1)
panel3avgred = panel3avgred.mean(axis=1)
panel3avgblue = panel3avgblue.mean(axis=1)
panel1avgred1h = panel1avgred1h.mean(axis=1)
panel1avgblue1h = panel1avgblue1h.mean(axis=1)

panel1avgred[12:20] = np.random.normal(loc=panel1avgred[7],scale=np.std(panel1avgred[6:11]),size=(8,))
panel1avgblue[12:20] = np.random.normal(loc=panel1avgblue[7],scale=np.std(panel1avgblue[6:11]),size=(8,))
panel2avgred[12:20] = np.random.normal(loc=panel2avgred[7],scale=np.std(panel2avgred[6:11]),size=(8,))
panel2avgblue[12:20] = np.random.normal(loc=panel2avgblue[7],scale=np.std(panel2avgblue[6:11]),size=(8,))
panel3avgred[12:20] = np.random.normal(loc=panel3avgred[7],scale=np.std(panel3avgred[6:11]),size=(8,))
panel3avgblue[12:20] = np.random.normal(loc=panel3avgblue[7],scale=np.std(panel3avgblue[6:11]),size=(8,))
panel1avgred1h[0:7*24]= np.random.normal(loc=panel1avgred[2],scale=np.std(panel1avgred[0:6]),size=(len(panel1avgred1h[0:7*24]),))
panel1avgred1h[7*24:len(panel1avgred1h+1)]= np.random.normal(loc=panel1avgred[7],scale=np.std(panel1avgred[6:11]),size=(len(panel1avgred1h[7*24:len(panel1avgred1h+1)]),))
panel1avgblue1h[0:7*24]= np.random.normal(loc=panel1avgblue[2],scale=np.std(panel1avgblue[0:6]),size=(len(panel1avgblue1h[0:7*24]),))
panel1avgblue1h[7*24:len(panel1avgblue1h+1)]= np.random.normal(loc=panel1avgblue[7],scale=np.std(panel1avgblue[6:11]),size=(len(panel1avgblue1h[7*24:len(panel1avgblue1h+1)]),))

panel1avgred = panel1avgred*18/24
panel1avgblue = panel1avgblue*18/24
panel2avgred = panel2avgred*18/24
panel2avgblue = panel2avgblue*18/24
panel3avgred = panel3avgred*18/24
panel3avgblue = panel3avgblue*18/24
panel1avgred1h = panel1avgred1h*18/24
panel1avgblue1h = panel1avgblue1h*18/24

avgarea = 7.5

lnarea1 = []
lnarea2 = []
lnarea3 = []
lnarea1_1h = []

for i in range(0,20):
    lnarea1.append(np.log(panel1area['panel1areaK'][i]-avgarea))
    lnarea2.append(np.log(panel2area['panel2areaK'][i]-avgarea))
    lnarea3.append(np.log(panel3area['panel3areaK'][i]-avgarea))
    
for i in range(0,24*10):
    lnarea1_1h.append(np.log(panel1area1h['panel1area'][i]-avgarea))    

time = list(range(1,21))
 
lnarea1 = pd.DataFrame({'lnarea':lnarea1,'avgredppfd':panel1avgred,'avgblueppfd':panel1avgblue,'ec':alldata['ph_3'],'ph':alldata['ec_3'],'time':time})  
lnarea2 = pd.DataFrame({'lnarea':lnarea2,'avgredppfd':panel2avgred,'avgblueppfd':panel2avgblue,'ec':alldata['ph_3'],'ph':alldata['ec_3'],'time':time})  
lnarea3 = pd.DataFrame({'lnarea':lnarea3,'avgredppfd':panel3avgred,'avgblueppfd':panel3avgblue,'ec':alldata['ph_3'],'ph':alldata['ec_3'],'time':time})  
lnarea1_1h = pd.DataFrame({'lnarea':lnarea1_1h,'avgredppfd':panel1avgred1h[0:24*10],'avgblueppfd':panel1avgblue1h[0:24*10],'ec':alldata1h['ph_3'][0:24*10],'ph':alldata1h['ec_3'][0:24*10],'time':[1+x/24 for x in range(0,24*10)]})  
lnarea1_1h = lnarea1_1h.bfill()



lnarea1 = lnarea1.ix[1:16,:]
lnarea2 = lnarea2.ix[1:16,:]
lnarea3 = lnarea3.ix[1:16,:]

#### adding data from 1st batch

numday= list(range(5,15))
area11 = [45.86,49.71,57.33,67.24,77.48,91.18,117.15,142.03,184.35,246.55] ####start 7 july (3rd actual)
lnarea11 = []


for i in range(0,10):
   lnarea11.append(np.log(area11[i]-avgarea)/(i+5))

L = list(range(0,5))

panel11avgred = [83.36*13/24 for i in L] + [83.36*13/24/2*3 for i in L]
panel11avgblue = [21.75*13/24 for i in L] + [21.75*13/24/2*3  for i in L]
ec = [1762,1748,1734,1720,1706,1692,1678,1664,1651,1637]
ph = [6.28,6.225,6.17,6.115,6.06,6.005,5.95,5.895,5.84,5.785]
lnarea11 = pd.DataFrame({'lnarea':lnarea11,'avgredppfd':panel11avgred,'avgblueppfd':panel11avgblue,'ec':ec,'ph':ph,'time':list(range(5,15))}) 

growthrate = pd.concat([lnarea1,lnarea2,lnarea3,lnarea11,lnarea1_1h],axis=0)


#ppfdred_mean = np.mean(growthrate['avgredppfd'])
#ppfdred_std = np.std(growthrate['avgredppfd'])
#ppfdblue_mean = np.mean(growthrate['avgblueppfd'])
#ppfdblue_std = np.std(growthrate['avgblueppfd'])
#ph_mean = np.mean(growthrate['ph'])
#ph_std = np.std(growthrate['ph'])
#ec_mean = np.mean(growthrate['ec'])
#ec_std = np.std(growthrate['ec'])
#
#growthrate['avgredppfd'] = (growthrate['avgredppfd'] - ppfdred_mean)/ppfdred_std
#growthrate['avgblueppfd'] = (growthrate['avgblueppfd'] - ppfdblue_mean)/ppfdblue_std
#growthrate['ph'] = (growthrate['ph'] - ph_mean)/ph_std
#growthrate['ec'] = (growthrate['ec'] - ph_mean)/ec_std

growthrate['avgredppfd'] = growthrate['avgredppfd']/160
growthrate['avgblueppfd'] = (growthrate['avgblueppfd'])/160
growthrate['ph'] = (growthrate['ph'])/14
growthrate['ec'] = (growthrate['ec'])/2200
growthrate['time'] = growthrate['time']/60

growthrate = growthrate.ix[:,['lnarea','avgredppfd','avgblueppfd','ph','ec','time']]  ##reorganize pairplots


growfactors = growthrate.ix[:,[1,2,5]]
leafarea = growthrate.ix[:,0]

import scipy as sp

def corrfunc(x, y, **kws):
    r, _ = sp.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
#
import seaborn as sns; sns.set(style="ticks", color_codes=True)
g = sns.pairplot(growthrate, kind="reg")
g.map_lower(corrfunc)
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

#import statsmodels.api as sm
#
#x= growfactors
#x= sm.add_constant(x)
#y = leafarea
#model = sm.OLS(y, x).fit_regularized(alpha=0.1, L1_wt=0)
#predictions = model.predict(x)
#model.summary()

from sklearn.linear_model import Ridge
score = []
score_i = []
for i in [x * 1 for x in range(0, 10)]:
    x= growfactors
    y = leafarea
    clf = Ridge(alpha=i, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=2, solver='auto', tol=0.001)
    clf.fit(x, y) 
    score_i.append(i)
    score.append(clf.score(x,y))
    print('alpha:',round(i,1),' coeffs:',clf.coef_,'intercept:',clf.intercept_,'R2 score:',clf.score(x,y))


dist = []
for i in range(0,len(score)):
    a = np.linalg.norm(np.array([score_i[len(score)-1],score[len(score)-1]])-np.array([score_i[0],score[0]]))
    b = np.linalg.norm(np.array([score_i[i],score[i]])-np.array([score_i[0],score[0]]))
    c=  np.linalg.norm(np.array([score_i[len(score)-1],score[len(score)-1]])-np.array([score_i[i],score[i]]))      
    s= 0.5*(a+b+c)
    area = (s*(s-a)*(s-b)*(s-c))**0.5
    dist.append(2*area/a)
print('best K: ',score_i[dist.index(max(dist))])
plt.figure(3)
plt.plot(score_i,score)

#coeffs = [  2.41152839 , 2.08074796 ,-0.11380985 ,-2.54685595,  4.12664251, 2.85591389438]
coeffs =[  5.31619626  , 3.55020806 , 17.44018821 ,  -3.06767567507]
##alpha: 0.1  coeffs: [ 6.18485348  4.16171218  2.40439953 -4.37113359  3.14483505] intercept: 0.398687720804 R2 score: 0.766585914269
##alpha: 0.2  coeffs: [ 5.36620658  4.45289938  2.18270786 -3.87171422  3.79043595] intercept: 0.41474767589 R2 score: 0.745121104726
##alpha: 0.1  coeffs: [ 5.37369407  4.53260909  3.81641431 -3.72005373  3.21968349] intercept: -0.415365156299 R2 score: 0.756950371453
#
#
lnarea1_est = []
lnarea2_est = []
lnarea3_est = []

for i in range(1,17):
#    lnarea1_est.append(np.exp(coeffs[0]*lnarea1['avgredppfd'][i]/160+coeffs[1]*lnarea1['avgblueppfd'][i]/160+coeffs[2]*lnarea1['ph'][i]/14+coeffs[3]*lnarea1['ec'][i]/2200+coeffs[4]*i/60+coeffs[5])+avgarea)
#    lnarea2_est.append(np.exp(coeffs[0]*lnarea2['avgredppfd'][i]/160+coeffs[1]*lnarea2['avgblueppfd'][i]/160+coeffs[2]*lnarea2['ph'][i]/14+coeffs[3]*lnarea2['ec'][i]/2200+coeffs[4]*i/60+coeffs[5])+avgarea)
#    lnarea3_est.append(np.exp(coeffs[0]*lnarea3['avgredppfd'][i]/160+coeffs[1]*lnarea3['avgblueppfd'][i]/160+coeffs[2]*lnarea3['ph'][i]/14+coeffs[3]*lnarea3['ec'][i]/2200+coeffs[4]*i/60+coeffs[5])+avgarea)  
    lnarea1_est.append(np.exp(coeffs[0]*lnarea1['avgredppfd'][i]/160+coeffs[1]*lnarea1['avgblueppfd'][i]/160+coeffs[4]*i/60+coeffs[5])+avgarea)
    lnarea2_est.append(np.exp(coeffs[0]*lnarea2['avgredppfd'][i]/160+coeffs[1]*lnarea2['avgblueppfd'][i]/160+coeffs[4]*i/60+coeffs[5])+avgarea)
    lnarea3_est.append(np.exp(coeffs[0]*lnarea3['avgredppfd'][i]/160+coeffs[1]*lnarea3['avgblueppfd'][i]/160+coeffs[4]*i/60+coeffs[5])+avgarea) 
        

readdate = []
for i in datearray:
    #readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m/%y %H:%M"))
    readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m"))

panel1area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel1k.csv', delimiter=',')
panel2area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel2k.csv', delimiter=',')
panel3area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel3k.csv', delimiter=',')


#Plotting area graphs

plt.figure(5)
plt.plot(list(range(1,21)),panel1area['panel1areaK'],'r',list(range(1,21)),panel2area['panel2areaK'],'b',list(range(1,21)),panel3area['panel3areaK'],'g',\
         list(range(1,17)),lnarea1_est,'y',list(range(1,17)),lnarea2_est,'c',list(range(1,17)),lnarea3_est,'k')
plt.xticks(list(range(1,21)),readdate[0:480:24])
plt.legend(['top','mid','bot','top_est','mid_est','bot_est'])