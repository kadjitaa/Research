# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:01:11 2022

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
import os, sys
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *



    
###############################################################################
###Setting Directory and Params
###############################################################################
data_directory   = r'C:\Users\kasum\Dropbox\ADn_Project'
info             = pd.read_excel(os.path.join(data_directory,'data_sheet.xlsx')) #directory to file with all exp data info

exp='standard'
cond1='light'
strains=['wt','rd1','gnat']

master_hdx=[]
master_nhdx=[]
wake1=[]
wake2=[]
for j,strain in enumerate(strains):
    idx2=[] #index for all the rows that meet cond1 and 2
    for i in range(len(info)):
        if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #:
            if cond1 in (info.iloc[i,:].values):
                idx2.append(i) 
    print(idx2)    
    
    for x,s in enumerate(idx2):
        path=info.dir[s].replace('\\',"/")
        print(str(x+1)+'/'+str(len(idx2)))
    
      
        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        episodes = info.filter(like='T').loc[s] #.dropna()
            
        if episodes.values[0]=='sleep' or 'NaN': 
            events=np.where(episodes.values==cond1)[0][0]-1

        
        else:
            events=np.where(episodes.values==cond1)[0][0]

          
        spikes, shank                       = loadSpikeData(path)
        n_channels, fs, shank_to_channel   = loadXML(path)
        position                            = loadPosition(path, events, episodes)
        wake_ep                             = loadEpoch(path, 'wake', episodes)
        #sleep_ep                            =loadEpoch(path, 'sleep', episodes)
        ep1=nts.IntervalSet(start=wake_ep.loc[events,'start'], end =wake_ep.loc[events,'start']+3e+8)
        ep2=nts.IntervalSet(start=wake_ep.loc[events,'start']+3e+8, end =wake_ep.loc[events,'start']+6e+8)
        
        
        ep=nts.IntervalSet(start=wake_ep.loc[events,'start'], end =wake_ep.loc[events,'start']+6e+8)
        tcurv_light = computeAngularTuningCurves(spikes,position['ry'],ep,60)
        hds,stats=findHDCells_GV(tcurv_light); #list of HD cells based on light condidtion
    
   
        hds_idx=[str(s)+'_'+str(k) for k in hds]
        nhds_idx=[str(s)+'_'+str(k) for k in spikes.keys() if k not in hds ]
        
        
        master_hdx.extend(hds_idx)
        master_nhdx.extend(nhds_idx)
        
        
        auto1,_=compute_AutoCorrs(spikes, ep1,spikes.keys(), 1, 200)
        auto2,_=compute_AutoCorrs(spikes, ep2,spikes.keys(), 1, 200)
        index 						= [str(s)+'_'+str(k) for k in spikes]
        auto1.columns 				= pd.Index(index)
        auto2.columns               = pd.Index(index)
        wake1.append(auto1)
        wake2.append(auto2)

dataset={'data':[wake1,wake2],'hd_labels':master_hdx, 'nhd_labels':master_nhdx}