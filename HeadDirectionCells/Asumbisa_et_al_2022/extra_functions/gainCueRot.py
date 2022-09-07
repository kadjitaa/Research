# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 23:04:53 2022

@author: kasum
"""


import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from functions import *
from pycircstat.descriptive import mean as circmean
###############################################################################
###Setting Directory and Params
###############################################################################
data_directory   = r'C:\Users\kasum\Dropbox\ADn_Project' #win

info             = pd.read_excel(os.path.join(data_directory,'data_sheet.xlsx')) #directory to file with all exp data info

exp='standard'
cond1='cueA'
cond2='cueB'

strains=['wt'] #you can equally specify the mouse you want to look at

#index for all the rows that meet cond1 and 2
for strain in strains:
    idx2=[]
    for i in range(len(info)):
        if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #:
            if cond1 in list(info.iloc[i,:].values) and cond2 in list(info.iloc[i,:].values):
                idx2.append(i)
                
#################################################################################
###Preselect Rows of Interest for group analysis
#################################################################################
    
    ##############################################################################
    ###Within Animal Analysis
    ################################################################################
    ###Combined Datasets      
    
    all_circMean=pd.DataFrame(columns=([cond1,cond2]))
    all_standard=pd.DataFrame(columns=(['observed']))
    all_rots=pd.DataFrame(columns=(['predicted']))
    all_gtypes=pd.DataFrame(columns=['strain'])
    all_stats=[]
    
    head_r={}
    ###############################################################################
    ###Data Processing
    ##############################################################################   
    cue_ctrl=[]
    gain=[]

    for x,s in enumerate(idx2):
        path=info.dir[s].replace('\\',"/")
        
        ############################################################################################### 
        # LOADING DATA
        ###############################################################################################
        episodes = info.filter(like='T').loc[s]
        events  = list(np.where((episodes == cond1) | (episodes== cond2))[0].astype('str'))
        
        spikes, shank                       = loadSpikeData(path)
        #n_channels, fs, shank_to_channel   = loadXML(path)
        position                            = loadPosition(path, events, episodes)
        wake_ep                             = loadEpoch(path, 'wake', episodes)
        
        ep1=nts.IntervalSet(start=wake_ep.loc[int(events[0])-1,'start'], end =wake_ep.loc[int(events[0])-1,'start']+6e+8)
        ep2=nts.IntervalSet(start=wake_ep.loc[int(events[-1])-1,'start'], end =wake_ep.loc[int(events[-1])-1,'start']+6e+8)
            
        tcurv_1 = computeAngularTuningCurves(spikes,position['ry'],ep1,60)
        tcurv_2 = computeAngularTuningCurves(spikes,position['ry'],ep2,60)
     
        hds1,_=findHDCells_GV(tcurv_1)
        hds2,_=findHDCells_GV(tcurv_2) 
        hds=list(set(hds1) or set(hds2))
        
        circ_mean,_ = computeCircularStats([ep1,ep2],spikes,position['ry'],[cond1,cond2],hds)   

        cond3=deg2rad(info.rot_ang[s]) # the absolute change in cue ang
        #cond4=info.rot_dir[s]  #CW or CCW cue rot
        gtype=info.genotype[s]
        
        h1=[]##control
        h2=[]##Gain
        for i in range(len(circ_mean)): 
            a=circ_mean.iloc[i,0]
            b=circ_mean.iloc[i,1]

        #compute the shortest distance between the 2 angles
            distance = (b - a) % (2*pi) 
            if distance < - np.pi:
                distance += (2*pi);
            elif distance > deg2rad(179):
                distance -= (2*pi);
                
            h1.append(abs(distance)-cond3) # diff of PFD1,2 and extent of cue rot.
            h2.append(abs(distance)/abs(cond3))
            
        cue_ctrl.extend(h1)
        gain.extend(h2)
 
      
    if strain=='wt':
        wt=[ cue_ctrl,gain]

    elif strain=='rd1':
        rd1=[ cue_ctrl,gain]
    elif strain=='gnat':
        gn=[ cue_ctrl,gain]
