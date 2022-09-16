# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:36:19 2022

@author: Asumbisa
"""
from wrappers import *
from functions import *
import glob, os 

###############################################################
# PARAMETERS
###############################################################
data_directory=r'D:\Asumbisa_et_al_22_Dataset\floorRot_sessions\wt\KA84-201128' #session directory
all_files = glob.glob(os.path.join(data_directory, "*.csv"))

#Find all epochs in Epoch_TS.csv
eps=[]
for i in range(len(all_files)):
    if all_files[i].split('_')[-1].split('.')[0] != 'TS':
        eps.append(int(all_files[i].split('_')[-1].split('.')[0]))

if 0 in eps:
    episodes=['wake']*len(eps)
else:
    episodes=['sleep']+(['wake']*len(eps))

#Generate index of wake epochs        
events=[i for i,x in enumerate(episodes) if x!='sleep']    

#Extract spikes, position data and wake epochs
n_analogin_channels = 2  #number of analogin channels open
channel_optitrack=1 #index of TTL channel
spikes= loadSpikeData(data_directory) #loads spikes
n_channels, fs = loadXML(data_directory)  #channel mappings 
position= loadPosition(data_directory,events,episodes,n_analogin_channels,channel_optitrack) #loads position
wake_ep=loadEpoch(data_directory,'wake',episodes) #loads time intervals for wake trial epochs

##################################################################
# PLOT: Tuning curves by trial
##################################################################
cols=int((len(spikes.keys()))/4)+1
for i in range(len(events)):
    ep=nts.IntervalSet(start=wake_ep.loc[i,'start'], end=wake_ep.loc[i,'end'])
    tc=computeAngularTuningCurves(spikes,position['ry'],ep,60)
    plt.figure()
    plt.suptitle('Trial '+ str(i+1))
    for x in spikes.keys():
        subplot(4,cols,1+x, projection='polar')
        plot(tc[x])
