# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:50:34 2022

@author: kasum
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn import preprocessing
from sklearn.cluster import KMeans,MiniBatchKMeans
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


data2=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\AutoCorrs_StandardL_ver2.npy',allow_pickle=True).item()

wake1=data2['data'][0]
wake2=data2['data'][1]

wak1 = pd.concat(wake1, 1)
wak2 = pd.concat(wake2, 1)

# 1. starting at 2
autocorr_wak1 = wak1.loc[0.5:]
autocorr_wak2 = wak2.loc[0.5:]

# # 4. gauss filt
autocorr_wak1 = autocorr_wak1.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.5)
autocorr_wak2 = autocorr_wak2.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.5)

autocorr_wak1 = autocorr_wak1[3:200]
autocorr_wak2 = autocorr_wak2[3:200]

# 6 combining all 
neurons = np.intersect1d(autocorr_wak1.columns, autocorr_wak2.columns)

#neurons=autocorr_wak1.columns
data = np.hstack([autocorr_wak1[neurons].values.T,autocorr_wak2[neurons].values.T])
df=pd.DataFrame(index=neurons,data=data)

cell_id=[]
for i in df.index:
    if i in data2['hd_labels']:
        cell_id.extend([1])
    else:
        cell_id.extend([0])

df.index=cell_id
hd_df=df[df.index==1][:153] #HD cells
nhd_df=df[df.index==0][:153] #non-HD cells
df1=pd.concat((hd_df,nhd_df),axis=0)

######################################################################################################
## XGBOOST
######################################################################################################
#split into train test
y=df1.index
X=df1.values

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0,test_size=0.2)

#Feature scaling
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#Evaluate me
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)



#################################################################################
## OSN DATA
#################################################################################
dataOSN=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\AutoCorrs_OSN_ver2.npy',allow_pickle=True).item()
################################################################################################################

wakeOSN1=dataOSN['data'][0]
wakeOSN2=dataOSN['data'][1]

wakOSN1 = pd.concat(wakeOSN1, 1)
wakOSN2 = pd.concat(wakeOSN2, 1)

# 1. starting at 2
autocorrOSN1 = wakOSN1.loc[0.5:]
autocorrOSN2 = wakOSN2.loc[0.5:]

# # 4. gauss filt
autocorrOSN1 = autocorrOSN1.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.5) #sd==2.5 perfect-
autocorrOSN2 = autocorrOSN2.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.5)

autocorrOSN1 = autocorrOSN1[3:200]
autocorrOSN2 = autocorrOSN2[3:200]

# 6 combining all 
neuronsOSN = np.intersect1d(autocorrOSN2.columns, autocorrOSN2.columns)

#neurons=autocorr_wak1.columns
dataOSN1 = np.hstack([autocorrOSN1[neuronsOSN].values.T,autocorrOSN1[neuronsOSN].values.T])
dfOSN=pd.DataFrame(index=neuronsOSN, data=dataOSN1)

##### USE MODEL TO MAKE NEW PREDICTIONS
newData=sc_X.fit_transform(dfOSN.values)
labels_new=model.predict(newData)


########################### VALIDATING ###########################################
ct=0
for i in dataOSN['WTL_hds']:
    place=np.where(neuronsOSN==i)[0][0]
    pred_label=labels_new[place]
    if pred_label==1:
        ct+=1
TP=round(((ct/len(dataOSN['WTL_hds']))*100),2)
#correctly identified HD cells 93.3% , False negative=6.6%
print('True Positive: '+str(TP)+'%')

ct=0
for i in dataOSN['WTL_nhds']:
    place=np.where(neuronsOSN==i)[0][0]
    pred_label=labels_new[place]
    if pred_label==0:
        ct+=1
(ct/len(dataOSN['WTL_nhds']))*100
#True negative=90.6,  fasle positive==9.3%
TN=round(((ct/len(dataOSN['WTL_nhds']))*100),2)
print('True Negative: '+str(TN)+'%')




#################################################################################
## Standard WT Dark DATA
#################################################################################
dataDark=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\AutoCorrs_StandardDarkWT_ver1.npy',allow_pickle=True).item()
        
wakeOSN1=dataDark['data'][0]
wakeOSN2=dataDark['data'][1]

wakOSN1 = pd.concat(wakeOSN1, 1)
wakOSN2 = pd.concat(wakeOSN2, 1)

# 1. starting at 2
autocorrOSN1 = wakOSN1.loc[0.5:]
autocorrOSN2 = wakOSN2.loc[0.5:]

# # 4. gauss filt
autocorrOSN1 = autocorrOSN1.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.5) #sd==2.5 perfect-
autocorrOSN2 = autocorrOSN2.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 1.5)
autocorrOSN1 = autocorrOSN1[3:200]
autocorrOSN2 = autocorrOSN2[3:200]

# 6 combining all 
neuronsOSN = np.intersect1d(autocorrOSN2.columns, autocorrOSN2.columns)

#neurons=autocorr_wak1.columns
dataOSN1 = np.hstack([autocorrOSN1[neuronsOSN].values.T,autocorrOSN1[neuronsOSN].values.T])
dfOSN=pd.DataFrame(index=neuronsOSN, data=dataOSN1)


##### USE MODEL TO MAKE NEW PREDICTIONS
newData=sc_X.fit_transform(dfOSN.values)
labels_new=model.predict(newData)

