# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:57:22 2022

@author: kasum
"""
from sklearn.datasets import load_boston
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
data = np.hstack([autocorr_wak1[neurons].values.T,autocorr_wak2[neurons].values.T]) #autocorrs from 1st and second half stacked together to rep cells activity
df=pd.DataFrame(index=neurons,data=data)



#############################################################################################################################
### XGB Excluding WT Dark Sessions in Training Data [aim is to hide data from classifier for later analysis]
############################################################################################################################
#needed if some sessions must be excluded in the model to avoid data leakage- based on eventual model use case .
# idx2=[17, 22, 154, 159, 178, 200] #paired WT light-Dark Standard sessions

# wt_paired_sess=[]
# for x,s in enumerate(idx2):
#     for i,cell in enumerate(neurons):
#         cell_ids=cell.split('_')[0]
#         if str(s)==cell_ids:
#             wt_paired_sess.append(i)


# wt_light_paired=df.iloc[wt_paired_sess,:]

# df.iloc[wt_paired_sess,:]=None # Dropped all WT paired L and Dark Sessions

####Assign pre-defined labels###################################################
cell_id=[]
for i in df.index:
    if i in data2['hd_labels']:
        cell_id.extend([1])
    else:
        cell_id.extend([0])

df.index=cell_id

df2=df.dropna() #remove all sessions from paired light-dark due to future analysis


nhd_df=df2[df2.index==0] #non-HD cells   
hd_df=df2[df2.index==1] #non-HD cells  



np.random.seed(11)
rand_rows=np.random.choice(np.arange(len(hd_df.index)),len(nhd_df),replace=False)
hd_df2=hd_df.iloc[rand_rows,:]

df1=pd.concat((hd_df2,nhd_df),axis=0)

######################################################################################################
## Train XGBOOST Model
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
model = XGBClassifier(objective='binary:logistic',use_label_encoder=False,eval_metric='logloss')
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
f_score=f1_score(y_test,y_pred)*100
print(f_score)

