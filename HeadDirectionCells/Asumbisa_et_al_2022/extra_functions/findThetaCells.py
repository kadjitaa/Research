# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 22:09:31 2022

@author: Asumbisa
"""
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter
from functions import *



def compute_AutoCorrs(spks, ep,hdIdx, binsize = 5, nbins = 400):
    # First let's prepare a pandas dataframe to receive the data
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2    
    autocorrs = pd.DataFrame(index = times, columns = hdIdx)
    firing_rates = pd.Series(index = hdIdx)

    # Now we can iterate over the dictionnary of spikes
    for i,x in enumerate(hdIdx):
        # First we extract the time of spikes in ms during wake
        spk_time = spks[x].restrict(ep).as_units('ms').index.values
        # Calling the crossCorr function
        autocorrs[x] = crossCorr(spk_time, spk_time, binsize, nbins)
        # Computing the mean firing rate
        firing_rates[x] = len(spk_time)/ep.tot_length('s')

    # We can divide the autocorrs by the firing_rates
    autocorrs = autocorrs / firing_rates

    # And don't forget to replace the 0 ms for 0
    autocorrs.loc[0] = 0.0
    return autocorrs, firing_rates



def thetaCells(spikes,ep,cell_ids):
    '''
    Parameters
    ----------
    spikes : dict of spikes
    ep : nts.IntervalSet
        The epoch/session of interest.
    cell_ids : list or array
        The cells you want to analyse (if all cells, set cell_ids = spikes.keys().
    nbins : int, optional
        Default is 200.

    Returns
    -------
    theta_mod : list of theta modulated cells
    '''
    auto,_=compute_AutoCorrs(spikes,ep,cell_ids,nbins=200)
 
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
        peaks_pro,_=scipy.signal.find_peaks(yf1[:13],prominence=0.018)
        if peaks_pro.size>0 and any(peaks_pro <13):
            theta_mod.append(i)
        else:
            pass
    return theta_mod

