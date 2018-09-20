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

panel1area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel1k_4h.csv', delimiter=',')
panel2area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel2k_4h.csv', delimiter=',')
panel3area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel3k_4h.csv', delimiter=',')
alldata = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//allpanel_4h.csv',delimiter=',')
alldata_1 = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//allpanel_hr_19jun_23jul.csv',delimiter=',')
panel1area_1 = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel.csv', delimiter=',')
alldata = alldata.ffill()
alldata = alldata.bfill()
alldata_1 = alldata_1.bfill()
alldata_1 = alldata_1.ffill()

panel1avgred = alldata.ix[:,['ppfdred1','ppfdred2','ppfdred3','ppfdred4','ppfdred5','ppfdred6','ppfdred7']]
panel1avgblue = alldata.ix[:,['ppfdblue1','ppfdblue2','ppfdblue3','ppfdblue4','ppfdblue5','ppfdblue6','ppfdblue7']]
panel2avgred = alldata.ix[:,['ppfdred1_2','ppfdred2_2','ppfdred3_2','ppfdred4_2','ppfdred5_2','ppfdred6_2','ppfdred7_2']]
panel2avgblue = alldata.ix[:,['ppfdblue1_2','ppfdblue2_2','ppfdblue3_2','ppfdblue4_2','ppfdblue5_2','ppfdblue6_2','ppfdblue7_2']]
panel3avgred = alldata.ix[:,['ppfdred1_3','ppfdred2_3','ppfdred3_3','ppfdred4_3','ppfdred5_3','ppfdred6_3','ppfdred7_3']]
panel3avgblue = alldata.ix[:,['ppfdblue1_3','ppfdblue2_3','ppfdblue3_3','ppfdblue4_3','ppfdblue5_3','ppfdblue6_3','ppfdblue7_3']]
panel1avgred_1 = alldata_1.ix[:,['ppfdred1','ppfdred2','ppfdred3','ppfdred4','ppfdred5','ppfdred6','ppfdred7']]
panel1avgblue_1 = alldata_1.ix[:,['ppfdblue1','ppfdblue2','ppfdblue3','ppfdblue4','ppfdblue5','ppfdblue6','ppfdblue7']]

panel1avgred = panel1avgred.mean(axis=1)
panel1avgblue = panel1avgblue.mean(axis=1)
panel2avgred = panel2avgred.mean(axis=1)
panel2avgblue = panel3avgblue.mean(axis=1)
panel3avgred = panel3avgred.mean(axis=1)
panel3avgblue = panel3avgblue.mean(axis=1)
panel1avgred_1 = panel1avgred_1.mean(axis=1)[432:649][::4].reset_index(drop=True)
panel1avgblue_1 = panel1avgblue_1.mean(axis=1)[432:649][::4].reset_index(drop=True)

###for panel1
panel1avgred[0:34] = np.random.normal(loc=panel1avgred[4],scale=np.std(panel1avgred[3:7]),size=(len(range(0,34)),))
panel1avgred[34:120] = np.random.normal(loc=panel1avgred[40],scale=np.std(panel1avgred[39:43]),size=(len(range(34,120)),))
panel1avgblue[0:34] = np.random.normal(loc=panel1avgblue[4],scale=np.std(panel1avgblue[3:7]),size=(len(range(0,34)),))
panel1avgblue[34:120] = np.random.normal(loc=panel1avgblue[40],scale=np.std(panel1avgblue[39:43]),size=(len(range(34,120)),))
###for panel2
panel2avgred[0:39] = np.random.normal(loc=panel2avgred[4],scale=np.std(panel2avgred[3:7]),size=(len(range(0,39)),))
panel2avgred[39:120] = np.random.normal(loc=panel2avgred[40],scale=np.std(panel2avgred[39:43]),size=(len(range(39,120)),))
panel2avgblue[0:40] = np.random.normal(loc=panel2avgblue[4],scale=np.std(panel2avgblue[3:9]),size=(len(range(0,40)),))
panel2avgblue[40:120] = np.random.normal(loc=panel2avgblue[40],scale=np.std(panel2avgblue[45:51]),size=(len(range(40,120)),))
###for panel3
panel3avgred[0:40] = np.random.normal(loc=panel3avgred[4],scale=np.std(panel3avgred[9:14]),size=(len(range(0,40)),))
panel3avgred[40:120] = np.random.normal(loc=panel3avgred[49],scale=np.std(panel3avgred[45:51]),size=(len(range(40,120)),))
panel3avgblue[0:40] = np.random.normal(loc=panel3avgblue[4],scale=np.std(panel3avgblue[9:14]),size=(len(range(0,40)),))
panel3avgblue[40:120] = np.random.normal(loc=panel3avgblue[40],scale=np.std(panel3avgblue[45:51]),size=(len(range(40,120)),))
###for panel1 first batch
panel1avgred_1[0:29] = np.random.normal(loc=panel1avgred_1[4],scale=np.std(panel1avgred_1[3:6]),size=(len(range(0,29)),))
panel1avgred_1[29:55] = np.random.normal(loc=panel1avgred_1[34],scale=np.std(panel1avgred_1[33:38]),size=(len(range(29,55)),))
panel1avgblue_1[0:29] = np.random.normal(loc=panel1avgblue_1[4],scale=np.std(panel1avgblue_1[3:6]),size=(len(range(0,29)),))
panel1avgblue_1[29:55] = np.random.normal(loc=panel1avgblue_1[34],scale=np.std(panel1avgblue_1[33:38]),size=(len(range(29,55)),))

####light intensity
panel1avgred = panel1avgred*18/24
panel1avgblue = panel1avgblue*18/24
panel2avgred = panel2avgred*18/24
panel2avgblue = panel2avgblue*18/24
panel3avgred = panel3avgred*18/24
panel3avgblue = panel3avgblue*18/24
panel1avgred_1 = panel1avgred_1*13/24
panel1avgblue_1 = panel1avgblue_1*13/24

avgarea = 7.5


panel1area_1 = panel1area_1[97:314][::4].reset_index(drop=True)

panel1area_1 = (16*panel1area_1-(2.5*2.5*np.pi))/15

lnarea1 = (np.log(panel1area['panel1areaK']-avgarea))
lnarea2 = (np.log(panel2area['panel2areaK']-avgarea))
lnarea3 = (np.log(panel3area['panel3areaK']-avgarea))

lnarea1b1 = pd.Series(np.log(panel1area_1-avgarea).values.reshape(55,))

alldata_1 = alldata_1[432:649][::4].reset_index(drop=True)

lnarea1 = pd.DataFrame({'lnarea':lnarea1,'avgredppfd':panel1avgred,'avgblueppfd':panel1avgblue,'ec':alldata['ph_3'],'ph':alldata['ec_3'],'time':[x*4/24 for x in list(range(1,121))]})  
lnarea2 = pd.DataFrame({'lnarea':lnarea2,'avgredppfd':panel2avgred,'avgblueppfd':panel2avgblue,'ec':alldata['ph_3'],'ph':alldata['ec_3'],'time':[x*4/24 for x in list(range(1,121))]})  
lnarea3 = pd.DataFrame({'lnarea':lnarea3,'avgredppfd':panel3avgred,'avgblueppfd':panel3avgblue,'ec':alldata['ph_3'],'ph':alldata['ec_3'],'time':[x*4/24 for x in list(range(1,121))]})  
lnarea1b1 = pd.DataFrame({'lnarea': lnarea1b1 ,'avgredppfd':panel1avgred_1,'avgblueppfd':panel1avgblue_1,'ec':alldata_1['ph_3'],'ph':alldata_1['ec_3'],'time':[x*4/24 for x in list(range(96-96,313-96,4))]})  

lnarea1 = lnarea1.ix[0:85,:]
lnarea2 = lnarea2.ix[0:85,:]
lnarea3 = lnarea3.ix[0:85,:]

growthrate = pd.concat([lnarea1,lnarea2,lnarea3,lnarea1b1],axis=0)


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

growthrate['avgredppfd'] = (growthrate['avgredppfd']/160)
growthrate['avgblueppfd'] = ((growthrate['avgblueppfd'])/160)
growthrate['ph'] = ((growthrate['ph'])/14)-1
growthrate['ec'][growthrate['ec']>5000] = 2000 
growthrate['ec'] = ((growthrate['ec'])/2200)-1
growthrate['time'] = growthrate['time']/30

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
g = sns.pairplot(growthrate.ix[:,[0,1,2,5]], kind="reg", diag_kind="kde",diag_kws=dict(shade=True))
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
#    clf = Lasso(alpha=i, copy_X=True, fit_intercept=True, max_iter=1000,
#   normalize=False, positive=True, precompute=False, random_state=None,
#   selection='cyclic', tol=0.0001, warm_start=False)
    
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


clf = Ridge(alpha= score_i[dist.index(max(dist))], copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=2, solver='auto', tol=0.001)

clf.fit(x, y) 

plt.figure(11)
plt.scatter(list(range(0,len(leafarea))),leafarea, marker="s",c='k',s=3)
plt.scatter(list(range(0,len(leafarea))),clf.predict(x), marker="s",c='r',s=3)
plt.legend(['Real DataPoints', 'Predicted Datapoints'])
plt.xlabel('Datapoints')    
plt.ylabel('ln of Est. Leaf Area')
plt.grid(b=True)
plt.title('Difference between Modelled vs. Actual Readings')





print('alpha:', score_i[dist.index(max(dist))],' coeffs:',clf.coef_,'intercept:',clf.intercept_,'R2 score:',clf.score(x,y))

coeffs = np.append(clf.coef_,clf.intercept_)

#
lnarea1_est = []
lnarea2_est = []
lnarea3_est = []

for i in range(0,86):
#    lnarea1_est.append(np.exp(coeffs[0]*(lnarea1['avgredppfd'][i]/160-1)+coeffs[1]*(lnarea1['avgblueppfd'][i]/160-1)+coeffs[2]*(lnarea1['ph'][i]/14-1)+\
#                              coeffs[3]*(lnarea1['ec'][i]/2200-1)+coeffs[4]*lnarea1['time'][i]/30+coeffs[5])+avgarea)
#    lnarea2_est.append(np.exp(coeffs[0]*(lnarea2['avgredppfd'][i]/160-1)+coeffs[1]*(lnarea2['avgblueppfd'][i]/160-1)+coeffs[2]*(lnarea2['ph'][i]/14-1)+\
#                              coeffs[3]*(lnarea2['ec'][i]/2200-1)+coeffs[4]*lnarea2['time'][i]/30+coeffs[5])+avgarea)
#    lnarea3_est.append(np.exp(coeffs[0]*(lnarea3['avgredppfd'][i]/160-1)+coeffs[1]*(lnarea3['avgblueppfd'][i]/160-1)+coeffs[2]*(lnarea3['ph'][i]/14-1)+\
#                              coeffs[3]*(lnarea3['ec'][i]/2200-1)+coeffs[4]*lnarea3['time'][i]/30+coeffs[5])+avgarea)
    lnarea1_est.append(np.exp(coeffs[0]*(lnarea1['avgredppfd'][i]/160)+coeffs[1]*(lnarea1['avgblueppfd'][i]/160)+coeffs[2]*lnarea1['time'][i]/30+coeffs[3])+avgarea)
    lnarea2_est.append(np.exp(coeffs[0]*(lnarea2['avgredppfd'][i]/160)+coeffs[1]*(lnarea2['avgblueppfd'][i]/160)+coeffs[2]*lnarea2['time'][i]/30+coeffs[3])+avgarea)
    lnarea3_est.append(np.exp(coeffs[0]*(lnarea3['avgredppfd'][i]/160)+coeffs[1]*(lnarea3['avgblueppfd'][i]/160)+coeffs[2]*lnarea3['time'][i]/30+coeffs[3])+avgarea)      

        

readdate = []
for i in datearray:
    #readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m/%y %H:%M"))
    readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m"))

panel1area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel1k.csv', delimiter=',')
panel2area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel2k.csv', delimiter=',')
panel3area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel3k.csv', delimiter=',')
panel1area_1 = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel.csv', delimiter=',')


#Plotting area graphs

plt.figure(5)
plt.plot(list(range(1,20*6,6)),panel1area['panel1areaK'],'r',list(range(1,20*6,6)),panel2area['panel2areaK'],'b',list(range(1,20*6,6)),panel3area['panel3areaK'],'g',\
         list(range(1,87)),lnarea1_est,'y',list(range(1,87)),lnarea2_est,'c',list(range(1,87)),lnarea3_est,'k')
plt.xticks(list(range(1,20*6,6)),readdate[0:480:24])
plt.legend(['top','mid','bot','top_est','mid_est','bot_est'])

#
panel1area_1_sub=panel1area_1[4*24:14*24][::24]

plt.figure(55)
plt.plot(list(range(1,16)),panel1area['panel1areaK'][0:15],'r',list(range(1,16)),panel2area['panel2areaK'][0:15],'b',list(range(1,16)),panel3area['panel3areaK'][0:15],'g',\
         list(range(5,15)),panel1area_1_sub['panel1area'][0:10],'k')
plt.xticks(list(range(1,16)),readdate[0:24*16:24])
plt.legend(['Top Panel 2nd Batch | Avg red/blue ppfd per day: 125.28/13.74 @ 18hrs on 6hrs off' ,'Mid Panel 2nd Batch | Avg red/blue ppfd per day: 106.71/40.69 @ 18hrs on 6hrs off',\
            'Bot Panel 2nd Batch | Avg red/blue ppfd per day: 101.8/40.98 @ 9hrs on 3hrs off',\
            'Top Panel 1st Batch | Avg red/blue ppfd per day: 47.56/20.34 @ 13hrs on 11hrs off'])
plt.xlabel('Date')
plt.ylabel('Est. Leaf Area (cm2)')
plt.grid(b=True)
plt.title('Estimated Leaf Area (cm2) for Experiment')

plt.figure(66)
plt.plot(list(range(0,161)),[np.exp(2.495*(i/160-1)) for i in list(range(0,161))],'r',list(range(0,161)),[np.exp(1.611*(i/160-1)) for i in list(range(0,161))],'b')
plt.legend(['Red Led','Blue Led'])
plt.xlabel('PPFD Settings (umol/m2/s)')
plt.ylabel('Multiplicative Factor')
plt.grid(b=True)
plt.title('Red and Blue PPFD Contribution towards Lettuce Growth')



