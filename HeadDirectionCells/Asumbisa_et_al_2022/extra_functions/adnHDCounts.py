# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 00:04:46 2021

@author: kasum
"""


import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
from pycircstat.descriptive import mean as circmean
import _pickle as cPickle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


###############################################################################
###Setting Directory and Params
###############################################################################
data_directory   = r'C:\Users\kasum\Dropbox\ADn_Project'
info             = pd.read_excel(os.path.join(data_directory,'data_sheet.xlsx')) #directory to file with all exp data info


strains=['wt','rd1','gnat']
cond1='light'; cond2='dark'; exp='OSN'

for strain in strains:
    idx2=[] #index for all the rows that meet cond1 and 2
    for i in range(len(info)):
        if np.any(info.iloc[i,:].str.contains(exp)) and np.any(info.iloc[i,:].str.contains(strain)) : #:
            if cond1 in (info.iloc[i,:].values) and cond2 in (info.iloc[i,:].values):
                idx2.append(i)           
    print(idx2)          
    #Combined DataFrames
    adn=[]
    cell_counts={}
    for x,s in enumerate(idx2):
        path=info.dir[s].replace('\\',"/")
        print(str(x)+'/'+str(len(idx2)))

      
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
        
        ep=nts.IntervalSet(start=wake_ep.loc[events,'start'], end =wake_ep.loc[events,'start']+6e+8)

        tcurv=computeAngularTuningCurves(spikes,position['ry'],ep,60)

        #Find HD cells
        hds1,stats=findHDCells_GV(tcurv,50)
        # nhds=list(set(spikes.keys())-set(hds1))
        
        #Remove HD cells that are theta modulated based on autocorrs
        auto,_=compute_AutoCorrs(spikes,ep,hds1,nbins=200) # compute autocorss
        theta_mod=[]
        for xx,i in enumerate(auto.columns)  :
            N = 200
            # sample spacing
            T = 0.005 #convert the time bins(ms) to seconds
            x = np.linspace(0.0, N*T, N, endpoint=False)
            y =sqrt(auto.loc[:,i].values)# np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
        
            yf = fft(y)
            xf = fftfreq(N, T)[3:N//2]#freq power for 1-3Hz excluded to 
            yf1=2.0/(N-3) * np.abs(yf[3:N//2])
            
            yf1=gaussian_filter(yf1,sigma=1)
            peaks_pro,_=scipy.signal.find_peaks(yf1[:13],prominence=0.01)#0.07 d=20
            if peaks_pro.size>0 and any(peaks_pro <13):
                theta_mod.append(i)
            else:
                pass
        hds=list(set(hds1)-set(theta_mod))
        
        
        shank_map=shank.flatten() 
        shank_spk_map=pd.DataFrame(data=shank_map,index=spikes.keys())
        ##################################################################################################################################
        # ANALYSIS
        #################################################################################################################################
        hd_shanks=np.unique([shank_spk_map.iloc[i,:].values[0] for i in hds if i in shank_spk_map.index.values.flatten()]) #shanks with HD cells
        
        ids=[] #all units on shanks with at least one HD cell
        for i in hd_shanks:
            ids.extend(np.where(shank_spk_map.values==i)[0])
        
        
        #Remove non-HD cells that are theta modulated from total ADn cell counts
        ids_hds_shank1=set(ids) - set(hds) #cells on same shank as HD cells but failed HD criteria
        
        auto2,_=compute_AutoCorrs(spikes,ep,ids_hds_shank1,nbins=200)
        
        tot_non_theta=len(ids)
        for xx,i in enumerate(auto2.columns)  :
            N = 200
            # sample spacing
            T = 0.005 #convert the time bins(ms) to seconds
            x = np.linspace(0.0, N*T, N, endpoint=False)
            y =sqrt(auto2.loc[:,i].values)# np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
        
            yf = fft(y)
            xf = fftfreq(N, T)[3:N//2]#freq power for 1-3Hz excluded to 
            yf1=2.0/(N-3) * np.abs(yf[3:N//2])
            
            yf1=gaussian_filter(yf1,sigma=1)
            peaks_pro,_=scipy.signal.find_peaks(yf1[:13],prominence=0.01)#0.07 d=20
            if peaks_pro.size>0 and any(peaks_pro <13):
                print(i)
                tot_non_theta-=1
            else:
                pass
        try:
            hd_perc=(len(hds)/tot_non_theta)*100
        except:
            hd_perc=np.nan
            
        adn.append(hd_perc)                      
        cell_counts[s]=[tot_non_theta,len(hds)]
        cell_counts[s]=len(hds)

  
    if strain=='wt':
        wt_d={'hd_%':adn,'counts':cell_counts}
        wt_d=cell_counts
    elif strain=='rd1':
        rd={'hd_%':adn,'counts':cell_counts}
       # rd_d=cell_counts
    elif strain=='gnat':
        gn={'hd_%':adn,'counts':cell_counts}
        