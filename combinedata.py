import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
import math
import pandas as pd


startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

datearray = list(range(startdate,enddate,60*60*4))

readdate = []
#
#for i in datearray:
#    #readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m/%y %H:%M"))
#    readdate.append(datetime.datetime.fromtimestamp(i).strftime("%d/%m"))
#
#panel1area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel1k.csv', delimiter=',')
#panel2area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel2k.csv', delimiter=',')
#panel3area = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel3k.csv', delimiter=',')
#
#'''
#Plotting area graphs
#'''
#plt.plot(list(range(1,21)),panel1area['panel1areaK'],'r',list(range(1,21)),panel2area['panel2areaK'],'b',list(range(1,21)),panel3area['panel3areaK'],'g')
#plt.xticks(list(range(1,21)),readdate[0:480:24])
#plt.legend(['top','mid','bot'])

os.chdir('C://Users//xxx//Desktop//smartfarm_data//data')
dirlist1 = glob.glob('f1*.csv')
dirlist1.sort(key=os.path.getmtime)
dirlist2 = glob.glob('f2*.csv')
dirlist2.sort(key=os.path.getmtime)
dirlist3 = glob.glob('f3*.csv')
dirlist3.sort(key=os.path.getmtime)

##### preprocessing for panel1
list1= []

for i in dirlist1:
    list1.append(pd.read_csv(i))
panel1 = pd.concat(list1,ignore_index =True)

panel1['unixtime'] = panel1['time'].apply(lambda x: round(datetime.datetime.strptime(x, '%d-%m-%y %H:%M').timestamp()))

for col in panel1.columns:
    try:
        panel1[col][panel1[col]=='None'] = None
        panel1[col] = panel1[col].astype(float)
    except:
        pass

panel1['temp_1'][panel1['temp_1']<18] = None
panel1['temp_2'][panel1['temp_2']<18] = None
panel1['temp_3'][panel1['temp_3']<18] = None
panel1['temp_a'][panel1['temp_a']<18] = None
panel1['humd_1'][panel1['humd_1']>100] = None
panel1['humd_2'][panel1['humd_2']>100] = None
panel1['humd_3'][panel1['humd_3']>100] = None
panel1['humd_a'][panel1['humd_a']>100] = None


panel1 = panel1.ffill()

##1st bath top 3july to 23 july
##1st batch top2 19jun to 8 jul
##1st batch top3 
##2nd batch all start 2 aug
#startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
#enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

datearray = list(range(startdate,enddate,60*60*4))

panel1_hr = pd.DataFrame(columns = ['unixtime','time']+list(panel1.columns[1:len(panel1.columns)-1]))

for k in range(0,len(datearray)):
    data_holder =  panel1[(panel1['unixtime']<datearray[k]+60*60) & (panel1['unixtime']>=datearray[k])]
    b = []
    b.append(datearray[k])
    b.append(datetime.datetime.fromtimestamp(datearray[k]).strftime("%d-%m-%y %H:%M"))
    columns = data_holder.columns[1:len(data_holder.columns)-1]
    print(k)
    b = b + list(np.mean(data_holder[columns].values,axis=0))
    
    panel1_hr.loc[k] = b

panel1_hr = panel1_hr.ffill()

panel1_hr.to_csv('C://Users//xxx//Desktop//smartfarm_data//panel1_4h.csv', index=False)

##### preprocessing for panel2
list2 = []

for i in dirlist2:
    list2.append(pd.read_csv(i))
panel2 = pd.concat(list2,ignore_index =True)
panel2['unixtime'] = panel2['time'].apply(lambda x: round(datetime.datetime.strptime(x, '%d-%m-%y %H:%M').timestamp()))
panel2.columns = [s+'_2' for s in panel2.columns.tolist()]


for col in panel2.columns[1:len(panel2.columns)-1]:
    try:
        panel2[col][panel2[col]=='None'] = None
        panel2[col] = panel2[col].astype(float)
    except:
        pass

panel2['temp_1_2'][panel2['temp_1_2']<18] = None
panel2['temp_2_2'][panel2['temp_2_2']<18] = None
panel2['temp_3_2'][panel2['temp_3_2']<18] = None
panel2['humd_1_2'][panel2['humd_1_2']>100] = None
panel2['humd_2_2'][panel2['humd_2_2']>100] = None
panel2['humd_3_2'][panel2['humd_3_2']>100] = None

panel2 = panel2.ffill()

#startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
#enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())


datearray = list(range(startdate,enddate,60*60*4))

panel2_hr = pd.DataFrame(columns = ['unixtime','time']+list(panel2.columns[1:len(panel2.columns)-1]))

for k in range(0,len(datearray)):
    data_holder =  panel2[(panel2['unixtime_2']<datearray[k]+60*60) & (panel2['unixtime_2']>=datearray[k])]
    b = []
    b.append(datearray[k])
    b.append(datetime.datetime.fromtimestamp(datearray[k]).strftime("%d-%m-%y %H:%M"))
    columns = data_holder.columns[1:len(data_holder.columns)-1]
    print(k)
    b = b + list(np.mean(data_holder[columns].values,axis=0))
    
    panel2_hr.loc[k] = b

panel2_hr = panel2_hr.ffill()
panel2_hr.to_csv('C://Users//xxx//Desktop//smartfarm_data//panel2_4h.csv', index=False)

##### preprocessing for panel3

list3 = []

for i in dirlist3:
    list3.append(pd.read_csv(i))
panel3 = pd.concat(list3,ignore_index =True)
panel3['unixtime'] = panel3['time'].apply(lambda x: round(datetime.datetime.strptime(x, '%d-%m-%y %H:%M').timestamp()))
panel3.columns = [s+'_3' for s in panel3.columns.tolist()]


for col in panel3.columns[1:len(panel3.columns)-1]:
    try:
        panel3[col][panel3[col]=='None'] = None
        panel3[col] = panel3[col].astype(float)
    except:
        pass

panel3['temp_1_3'][panel3['temp_1_3']<18] = None
panel3['temp_2_3'][panel3['temp_2_3']<18] = None
panel3['temp_3_3'][panel3['temp_3_3']<18] = None
panel3['humd_1_3'][panel3['humd_1_3']>100] = None
panel3['humd_2_3'][panel3['humd_2_3']>100] = None
panel3['humd_3_3'][panel3['humd_3_3']>100] = None

panel3 = panel3.ffill()

#startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
#enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

startdate = round(datetime.datetime(2018,8,2,0,0).timestamp())
enddate = round(datetime.datetime(2018,8,22,0,0).timestamp())

datearray = list(range(startdate,enddate,60*60*4))

panel3_hr = pd.DataFrame(columns = ['unixtime','time']+list(panel3.columns[1:len(panel3.columns)-1]))

for k in range(0,len(datearray)):
    data_holder =  panel3[(panel3['unixtime_3']<datearray[k]+60*60) & (panel3['unixtime_3']>=datearray[k])]
    b = []
    b.append(datearray[k])
    b.append(datetime.datetime.fromtimestamp(datearray[k]).strftime("%d-%m-%y %H:%M"))
    columns = data_holder.columns[1:len(data_holder.columns)-1]
    print(k)
    b = b + list(np.mean(data_holder[columns].values,axis=0))
    
    panel3_hr.loc[k] = b

panel3_hr = panel3_hr.ffill()
panel3_hr.to_csv('C://Users//xxx//Desktop//smartfarm_data//panel3_4h.csv', index=False)

##### combined all

panel1_hr = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel1_4h.csv', delimiter=',')
panel2_hr = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel2_4h.csv', delimiter=',')
panel3_hr = pd.read_csv('C://Users//xxx//Desktop//smartfarm_data//panel3_4h.csv', delimiter=',')

panel2_hr = panel2_hr[panel2_hr.columns[2:len(panel2_hr.columns)]]
panel3_hr = panel3_hr[panel3_hr.columns[2:len(panel3_hr.columns)]]

allpanel_hr = pd.concat([panel1_hr,panel2_hr,panel3_hr],axis=1)

allpanel_hr.to_csv('C://Users//xxx//Desktop//smartfarm_data//allpanel_4h.csv', index=False)

