#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:41:03 2019
"""

import numpy as np
from numba import jit
import pandas as pd
import neuroseries as nts
import sys
import scipy
from sklearn.manifold import Isomap
from matplotlib.colors import hsv_to_rgb
from pylab import *
from scipy.stats import circmean
from scipy.stats import circvar
from scipy.ndimage import gaussian_filter
from itertools import combinations
from pycircstat.descriptive import mean as circmean2
import astropy

'''
Utilities functions
Feel free to add your own
'''


def refineSleepFromAccel(acceleration, sleep_ep):
	vl = acceleration[0].restrict(sleep_ep)
	vl = vl.as_series().diff().abs().dropna()	
	a, _ = scipy.signal.find_peaks(vl, 0.025)
	peaks = nts.Tsd(vl.iloc[a])
	duration = np.diff(peaks.as_units('s').index.values)
	interval = nts.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[1:])

	newsleep_ep = interval.iloc[duration>15.0]
	newsleep_ep = newsleep_ep.reset_index(drop=True)
	newsleep_ep = newsleep_ep.merge_close_intervals(100000, time_units ='us')

	#newsleep_ep	= sleep_ep.intersect(newsleep_ep)

	return newsleep_ep




#########################################################
# CORRELATION
#########################################################
# @jit(nopython=True)

def pairs(tcurves):
    '''generates cell pairs based on angular difference in ascending order
    The input to the function is a dataframe of tuning curves'''
    
    pf=pd.DataFrame(index=[0],columns=tcurves.columns)
    for i in tcurves.columns:
        pf[i]=tcurves[i].idxmax()
        
    cells=list(combinations(tcurves.columns,2))
    ang_diff=pd.DataFrame(index=[0],columns=cells)
    for i,x in ang_diff.columns:
        unwrap_diff=pf[i].values-pf[x].values
        ang_diff[(i,x)]=abs(np.arctan2(np.sin(unwrap_diff),np.cos(unwrap_diff)))

    cell_pairs=ang_diff.T.sort_values(by=[0]).index
    return cell_pairs

def qwik_cc(cell_a,cell_b, spikes, ep):
    t1=spikes[cell_a].restrict(ep).as_units('ms')
    t1_t=t1.index.values
    
    t2=spikes[cell_b].restrict(ep).as_units('ms')
    t2_t=t2.index.values
    
    # Let's say you want to compute the autocorr with 10 ms bins
    binsize = 5
    # with 200 bins
    nbins = 400 #400
    
    autocorr_0 = crossCorr(t1_t, t2_t, binsize, nbins)
    
    # The corresponding times can be computed as follow 
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    
    # Let's make a time series
    autocorr_0 = pd.Series(index = times, data = autocorr_0)
    
    mean_fr_0 = len(t1)/ep.tot_length('s')
    autocorr_0 = autocorr_0 / mean_fr_0
    
    return autocorr_0

def compute_CrossCorrs(spks, ep, binsize=10, nbins = 2000):
    """
        
    """    
    neurons = list(spks.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2    
    cc = pd.DataFrame(index = times, columns = list(combinations(neurons, 2)))
        
    for i,j in cc.columns:        
        spk1 = spks[i].restrict(ep).as_units('ms').index.values
        spk2 = spks[j].restrict(ep).as_units('ms').index.values        
        tmp = crossCorr(spk1, spk2, binsize, nbins)        
        fr = len(spk2)/ep.tot_length('s')
        cc[(i,j)] = tmp/fr

    return cc


def crossCorr(t1, t2, binsize, nbins):
    ''' 
        Fast crossCorr 
    '''
    nt1 = len(t1)
    nt2 = len(t2)
    if np.floor(nbins/2)*2 == nbins:
        nbins = nbins+1

    m = -binsize*((nbins+1)/2)
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m+j*binsize

    w = ((nbins/2) * binsize)
    C = np.zeros(nbins)
    i2 = 1

    for i1 in range(nt1):
        lbound = t1[i1] - w
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2+1
        while i2 > 1 and t2[i2-1] > lbound:
            i2 = i2-1

        rbound = lbound
        l = i2
        for j in range(nbins):
            k = 0
            rbound = rbound+binsize
            while l < nt2 and t2[l] < rbound:
                l = l+1
                k = k+1

            C[j] += k

    # for j in range(nbins):
    # C[j] = C[j] / (nt1 * binsize)
    C = C/(nt1 * binsize/1000)

    return C

def crossCorr2(t1, t2, binsize, nbins):
    '''
        Slow crossCorr
    '''
    window = np.arange(-binsize*(nbins/2),binsize*(nbins/2)+2*binsize,binsize) - (binsize/2.)
    allcount = np.zeros(nbins+1)
    for e in t1:
        mwind = window + e
        # need to add a zero bin and an infinite bin in mwind
        mwind = np.array([-1.0] + list(mwind) + [np.max([t1.max(),t2.max()])+binsize])    
        index = np.digitize(t2, mwind)
        # index larger than 2 and lower than mwind.shape[0]-1
        # count each occurences 
        count = np.array([np.sum(index == i) for i in range(2,mwind.shape[0]-1)])
        allcount += np.array(count)
    allcount = allcount/(float(len(t1))*binsize / 1000)
    return allcount

def xcrossCorr_slow(t1, t2, binsize, nbins, nbiter, jitter, confInt):        
    times             = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    H0                 = crossCorr(t1, t2, binsize, nbins)    
    H1                 = np.zeros((nbiter,nbins+1))
    t2j                 = t2 + 2*jitter*(np.random.rand(nbiter, len(t2)) - 0.5)
    t2j             = np.sort(t2j, 1)
    for i in range(nbiter):            
        H1[i]         = crossCorr(t1, t2j[i], binsize, nbins)
    Hm                 = H1.mean(0)
    tmp             = np.sort(H1, 0)
    HeI             = tmp[int((1-confInt)/2*nbiter),:]
    HeS             = tmp[int((confInt + (1-confInt)/2)*nbiter)]
    Hstd             = np.std(tmp, 0)

    return (H0, Hm, HeI, HeS, Hstd, times)

def xcrossCorr_fast(t1, t2, binsize, nbins, nbiter, jitter, confInt):        
    times             = np.arange(0, binsize*(nbins*2+1), binsize) - (nbins*2*binsize)/2
    # need to do a cross-corr of double size to convolve after and avoid boundary effect
    H0                 = crossCorr(t1, t2, binsize, nbins*2)    
    window_size     = 2*jitter//binsize
    window             = np.ones(window_size)*(1/window_size)
    Hm                 = np.convolve(H0, window, 'same')
    Hstd            = np.sqrt(np.var(Hm))    
    HeI             = np.NaN
    HeS             = np.NaN    
    return (H0, Hm, HeI, HeS, Hstd, times)    

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

def compute_ISI(spks, ep, maxisi, nbins, log_=False):
	"""
	"""
	neurons = list(spks.keys())
	if log_:
		bins = np.linspace(np.log10(1), np.log10(maxisi), nbins)
	else:
		bins = np.linspace(0, maxisi, nbins)
		
	isi = pd.DataFrame(index =  bins[0:-1] + np.diff(bins)/2, columns = neurons)
	for i in neurons:
		tmp = []
		for j in ep.index:
			tmp.append(np.diff(spks[i].restrict(ep.loc[[j]]).as_units('ms').index.values))
		tmp = np.hstack(tmp)
		if log_:
			isi[i], _ = np.histogram(np.log10(tmp), bins)
		else:
			isi[i], _ = np.histogram(tmp, bins)

	return isi





def butter_bandpass(lowcut, highcut, fs, order=5):
	from scipy.signal import butter
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	from scipy.signal import lfilter
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y




#########################################################
# VARIOUS 3.456246178867965


def computeAngularTuningCurves_dat(spikes, angle, ep, nb_bins = 180, frequency = 120.0, bin_size = 100):
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))    
    tmp2             = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)    
    bin_size         = bin_size * 1000
    time_bins        = np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
    index             = np.digitize(tmp2.index.values, time_bins)
    tmp3             = tmp2.groupby(index).mean()
    tmp3.index         = time_bins[np.unique(index)-1]+bin_size/2
    tmp3             = nts.Tsd(tmp3)
    tmp4            = np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
    newangle         = nts.Tsd(t = tmp3.index.values, d = tmp3.values%(2*np.pi))
    velocity         = nts.Tsd(t=tmp3.index.values[1:], d = tmp4)
    velocity         = velocity.restrict(ep)    
    velo_spikes     = {}    
    #for k in spikes: velo_spikes[k]    = velocity.realign(spikes[k].restrict(ep))
    #bins_velocity    = np.array([velocity.min(), -2*np.pi/3, -np.pi/6, np.pi/6, 2*np.pi/3, velocity.max()+0.001])
    #idx_velocity     = {k:np.digitize(velo_spikes[k].values, bins_velocity)-1 for k in spikes}

    bins             = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves     = {i:pd.DataFrame(index = idx, columns = np.arange(len(spikes))) for i in range(3)}    

    for i,j in zip(range(3),range(0,6,2)):
        for k in spikes:
            spks             = spikes[k].restrict(ep)            
            #spks             = spks[idx_velocity[k] == j]
            angle_spike     = newangle.restrict(ep).realign(spks)
            spike_count, bin_edges = np.histogram(angle_spike, bins)
            #tmp             = newangle.loc[velocity.index[np.logical_and(velocity.values>bins_velocity[j], velocity.values<bins_velocity[j+1])]]
            occupancy, _     = np.histogram(tmp, bins)
            spike_count     = spike_count/occupancy    
            tuning_curves[i][k] = spike_count*(1/(bin_size*1e-6))

    return tuning_curves, velocity, bins_velocity

def computeFiringRates(spikes, epochs, tcs,name,hds):
    mean_frate = pd.DataFrame(index = np.arange(len(hds)), columns = name)
    peak_frate= pd.DataFrame(index = np.arange(len(hds)), columns = name)
    pfd= pd.DataFrame(index = np.arange(len(hds)), columns = name)
    
    for n, ep, tc in zip(name, epochs,tcs):
        for i,k in enumerate(hds):
            mean_frate.loc[i,n] = len(spikes[k].restrict(ep))/ep.tot_length('s') 
            peak_frate.loc[i,n]=tc[k].max()
            pfd.loc[i,n]=tc[k].idxmax(axis=0)
            #tcs=computeAngularTuningCurves(spikes, position['ry'], ep)
            #peak_frate.loc[i,n] =tcs[k].max()
            #pfd.loc[i,n]=tcs[k].idxmax(axis=0)      
    return mean_frate, peak_frate, pfd

def computeVectorLength(spikes,epochs,position, name,hds):
    rmean=pd.DataFrame(index = np.arange(len(hds)), columns = name)
    
    for n, ep in zip(name, epochs):
        for i,k in enumerate (hds):
            spk = spikes[k]
            spk = spk.restrict(ep)
            angle_spk = position.realign(spk)
            C = np.sum(np.cos(angle_spk.values))
            S = np.sum(np.sin(angle_spk.values))
            Rmean = np.sqrt(C**2  + S** 2) /len(angle_spk)
            rmean.loc[i,n]=Rmean
    return rmean
                   

def VectorLength(spikes,epochs,position,hds):
    rmean=pd.DataFrame(index = hds,columns=[0])  
    for k in hds:
        spk = spikes[k]
        spk = spk.restrict(epochs)
        angle_spk = position.realign(spk)
        C = np.sum(np.cos(angle_spk.values))
        S = np.sum(np.sin(angle_spk.values))
        Rmean = np.sqrt(C**2  + S** 2) /len(angle_spk)
        rmean.loc[k,0]=Rmean
    return rmean

def MutualInfo(spikes,ep,position,hds):
    I=pd.DataFrame(index=hds,columns=[0])
    for k in hds: 
        lamda_i=computeAngularTuningCurves(spikes,position,ep,60)[k].values
        #bins=computeAngularTuningCurves(spikes,position['ry'],ep,60)[i].index
        lamda=len(spikes[k].restrict(ep))/ep.tot_length('s') 
        pos=position.restrict(ep)
        bins=linspace(0,2*pi,60)
        occu,a=np.histogram(pos, bins)
        occu= occu/sum(occu)
        bits_spk=sum(occu*(lamda_i/lamda)*np.log2(lamda_i/lamda))
        I.loc[k,0]=bits_spk
    return I      


def computeAngularTuningCurves(spikes, angle, ep,nb_bins = 180, frequency = 120.0):

    bins             = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves     = pd.DataFrame(index = idx, columns = spikes.keys())    
    angle             = angle.restrict(ep)
    # Smoothing the angle here
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    tmp2             = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    angle            = nts.Tsd(tmp2%(2*np.pi))
    for k in spikes:
        spks             = spikes[k]
        # true_ep         = nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))        
        spks             = spks.restrict(ep)    
        angle_spike     = angle.restrict(ep).realign(spks)
        spike_count, bin_edges = np.histogram(angle_spike, bins)
        occupancy, _     = np.histogram(angle, bins)
        spike_count     = spike_count/occupancy        
        tuning_curves[k] = spike_count*frequency    
    
        tcurves = tuning_curves[k]
        padded     = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi),
                                                tcurves.index.values,
                                                tcurves.index.values+(2*np.pi))),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)        
        tuning_curves[k] = smoothed[tcurves.index]

    return tuning_curves





def computeAngularTuningCurves2(spikes, angle, ep, hds,nb_bins = 180, frequency = 120.0):

    bins             = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves     = pd.DataFrame(index = idx, columns = hds)    
    angle             = angle.restrict(ep)
    # Smoothing the angle here
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    tmp2             = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    angle            = nts.Tsd(tmp2%(2*np.pi))
    for k in hds:
        spks             = spikes[k]
        # true_ep         = nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))        
        spks             = spks.restrict(ep)    
        angle_spike     = angle.restrict(ep).realign(spks)
        spike_count, bin_edges = np.histogram(angle_spike, bins)
        occupancy, _     = np.histogram(angle, bins)
        spike_count     = spike_count/occupancy        
        tuning_curves[k] = spike_count*frequency    
    
        tcurves = tuning_curves[k]
        padded     = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi),
                                                tcurves.index.values,
                                                tcurves.index.values+(2*np.pi))),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)        
        tuning_curves[k] = smoothed[tcurves.index]

    return tuning_curves


def computeFrateAng(spikes, angle, ep, nb_bins = 180, frequency = 120.0):
    '''Computes the ang tcurves without normalising to occupancy.
    It will essentiall give you the total spike count for each angular position
    '''

    bins             = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves     = pd.DataFrame(index = idx, columns = np.arange(len(spikes)))    
    angle             = angle.restrict(ep)
    # Smoothing the angle here
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    tmp2             = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    angle            = nts.Tsd(tmp2%(2*np.pi))
    for k in spikes:
        spks             = spikes[k]
        # true_ep         = nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))        
        spks             = spks.restrict(ep)    
        angle_spike     = angle.restrict(ep).realign(spks)
        spike_count, bin_edges = np.histogram(angle_spike, bins)
        occupancy, _     = np.histogram(angle, bins)
        tuning_curves[k] = spike_count    

    return tuning_curves

def computeInfo(eps,spikes,position, name,hds):
    I=pd.DataFrame(index=np.arange(len(hds)),columns=name)
    for n, ep in zip(name, eps):
        for i,k in enumerate(hds): 
            lamda_i=computeAngularTuningCurves(spikes,position,ep,60)[k].values
            #bins=computeAngularTuningCurves(spikes,position['ry'],ep,60)[i].index
            lamda=len(spikes[k].restrict(ep))/ep.tot_length('s') 
            pos=position.restrict(ep)
            bins=linspace(0,2*pi,60)
            occu,a=np.histogram(pos, bins)
            occu= occu/sum(occu)
            bits_spk=sum(occu*(lamda_i/lamda)*np.log2(lamda_i/lamda))
            I.loc[i,n]=bits_spk
    return I


def computePlaceInfo(spikes, position, ep, nb_bins = 60, frequency = 120.0):
    Info=pd.DataFrame(index=spikes.keys(),columns=['bits/spk'])
    position_tsd = position.restrict(ep)
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)    
    for n in spikes:
        position_spike = position_tsd.realign(spikes[n].restrict(ep))
        spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
        occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
        mean_spike_count = spike_count/(occupancy+1)
        place_field = mean_spike_count*frequency    
        place_fields = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
        occus= occupancy/sum(occupancy)
        mFR=len(spikes[n].restrict(ep))/ep.tot_length('s')
        info=occus*(place_fields/mFR)*np.log2(place_fields/mFR)
        info=np.array(info)
        info[isnan(info)]=0 
        Info.loc[n]=sum(info)
    return Info



   
#Computes mutual information for a single session--OLD
def hd_info(tcurve,ep,spikes,position):
    I=pd.DataFrame(index=spikes.keys(),columns=['Ispk'])
    for i in spikes.keys():
        lamda_i=tcurve[i].values
        #bins=tcurve.index
        lamda=len(spikes[i].restrict(ep))/ep.tot_length('s')
        
        pos=position['ry'].restrict(ep)
        bins=linspace(0,2*pi,60)
        occu,a=np.histogram(pos, bins)
        occu= occu/sum(occu)
        bits_spk=sum(occu*(lamda_i/lamda)*np.log2(lamda_i/lamda))
        I.loc[i,'Ispk']=bits_spk
    return I 

def fisher_information(x, f):
    """ Compute Fisher Information over the tuning curves
        x : array of angular position
        f : firing rate
        return (angular position, fisher information)
    """
    fish = np.zeros(len(f)-1)
    slopes_ = []
    tmpf = np.hstack((f[-1],f,f[0:3]))
    binsize = x[1]-x[0]	
    tmpx = np.hstack((np.array([x[0]-binsize-(x.min()+(2*np.pi-x.max()))]),x,np.array([x[-1]+i*binsize+(x.min()+(2*np.pi-x.max())) for i in range(1,4)])))		
    for i in range(len(f)):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(tmpx[i:i+3], tmpf[i:i+3])
        slopes_.append(slope)
    fish = np.power(slopes_, 2)
    fish = fish/(f+1e-4)
    return (x, fish)             
        
def findHDCells_GV(tuning_curves, z = 50, p = 0.0001 , m = 1):
	"""
		Peak firing rate larger than 1
		and Rayleigh test p<0.001 & z > 100
	"""
	cond1 = tuning_curves.max()>m
	from pycircstat.tests import rayleigh
	stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
	for k in tuning_curves:
		stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
	cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
	tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]
	return tokeep, stat



def findHDCells_KA(tuning_curves,pos,ep,spikes, z = 35, corr=0.4, p = 0.0001 , m = 1):
    cond1 = tuning_curves.max()>m
    stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z','r'])
    ep1_start=ep.start[0]; ep1_end=ep.start[0]+(diff(ep)[0][0]/2)
    
    ep2_start=ep1_end; ep2_end=ep.end[0]
    
    ep1=nts.IntervalSet(ep1_start,ep1_end)
    ep2=nts.IntervalSet(ep2_start,ep2_end)
    tcurve1=computeAngularTuningCurves(spikes,pos,ep1,60)
    tcurve2=computeAngularTuningCurves(spikes,pos,ep2,60)
    from pycircstat.tests import rayleigh
    from astropy.stats import circcorrcoef

    for k in tuning_curves.columns:
        stat.loc[k,['pval','z']] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
        stat.loc[k,'r']=circcorrcoef(tcurve2[k].replace(nan,0).values,tcurve1[k].replace(nan,0).values)

    cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
    cond3 = abs(stat['r'])>=corr
    #tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]
    tokeep =np.where(cond1&cond2&cond3)
    return tokeep, stat


def findHDCells(tuning_curves,ep,spikes,position,cut_off=0.49):
    """
        Peak firing rate larger than 1
        and Rayleigh test p<0.001 & z > 100
    """
    cond1 = pd.DataFrame(tuning_curves.max()>1.0)
    angle = position.restrict(ep)
    
    from pycircstat.tests import rayleigh
    
    stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
    for k in tuning_curves:
        stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
        #stat.loc[k]=  rayleigh(position['ry'].restrict(ep).values , position['ry'].realign(spikes[k].restrict(ep)))
                
    
    rMean=pd.DataFrame(index=tuning_curves.columns, columns=['hd_score'])   
    for k in tuning_curves:  
        """computes the rayleigh vector length as hdScore. 
        """
        spk = spikes[k]
        spk = spk.restrict(ep)
        angle_spk = angle.realign(spk)
        C = np.sum(np.cos(angle_spk.values))
        S = np.sum(np.sin(angle_spk.values))
        Rmean = np.sqrt(C**2  + S** 2) /len(angle_spk)
        rMean.loc[k]=Rmean
        
    stat['hd_score']=rMean
    
    spatial_corr=stability(ep,spikes,position)
    stat['stability']=spatial_corr.spatial_corr
    
    cond2 = pd.DataFrame(np.logical_and(stat['pval']<0.001,stat['z']>15))
    cond3 = pd.DataFrame(rMean['hd_score']>=cut_off)
    cond4= pd.DataFrame(stat.stability>=0.65)
    '''To Do
    Add cond 4 and set it to any value greater than 0.4'''
    #cond4=pd.DataFrame(stat['hd_info'])
    tokeep=(cond1[0]==True) & (cond2[0]==True) & (cond4['stability']==True) & (cond3['hd_score']==True) #was excluded a lot of obvious HD cells
    stat['hd_cells']=tokeep
 #tuning_curves have been normalised by occupancy, unlike the rvector
#Rayleigh z test to test the null hypothesis that there is no sample mean direction  
    
    '''I=pd.DataFrame(index=spikes.keys(),columns=['Ispk'])
    for i in spikes.keys():
        lamda_i=tuning_curves[i].values
        bins=tuning_curves.index
        lamda=len(spikes[i].restrict(ep))/ep.tot_length('s')
        
        pos=position.restrict(ep)
        bins=linspace(0,2*pi,60)
        occu,a=np.histogram(pos, bins)
        occu= occu/sum(occu)
        bits_spk=sum(occu*(lamda_i/lamda)*np.log2(lamda_i/lamda))
        I.loc[i,'Ispk']=bits_spk
    stat['hd_info']=I'''
  
    return stat
# np.where(cond3)[0] to get the index of the true statements
    
def decodeHD(tuning_curves, spikes, ep,bin_size = 200, px = None):
    """
        See : Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
        tuning_curves: pd.DataFrame with angular position as index and columns as neuron
        spikes : dictionnary of spike times
        ep : nts.IntervalSet, the epochs for decoding
        bin_size : in ms (default:200ms)
        px : Occupancy. If None, px is uniform
    """        
    if len(ep) == 1:
        bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
    else:
        print("TODO, more than one epoch")
        sys.exit()
    
   # spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = spikes.keys())
    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = tuning_curves.columns)

    
    for i,k in enumerate(tuning_curves.columns):
        spks = spikes[k].restrict(ep).as_units('ms').index.values
        spike_counts[k], _ = np.histogram(spks, bins)
    
    print(spike_counts.columns.values)
    print(tuning_curves.columns.values)

    tcurves_array = tuning_curves.values
    spike_counts_array = spike_counts.values
    proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

    part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
    if px is not None:
        part2 = px
    else:
        part2 = np.ones(tuning_curves.shape[0])
        # part2 = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
    
    for i in range(len(proba_angle)):
        part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
        p = part1 * part2 * part3
        proba_angle[i] = p/p.sum() # Normalization process here

    proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)    
    proba_angle = proba_angle.astype('float')        
    decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
    return decoded, proba_angle

def makeBins(ep, bin_size=200): #the bin size is based on the bin size of the decoder
    bins_=  np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)      
    return bins_


def computePlaceFields(spikes, position, ep, nb_bins = 100, frequency = 120.0):
    place_fields = {}
    occus={}
    position_tsd = position.restrict(ep)
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)    
    for n in spikes:
        position_spike = position_tsd.realign(spikes[n].restrict(ep))
        spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
        occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
        mean_spike_count = spike_count/(occupancy+1)
        place_field = mean_spike_count*frequency    
        place_fields[n] = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
        occus[n]= occupancy/sum(occupancy)
    extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB
    return place_fields, extent

def computeOccupancy(position_tsd, nb_bins = 100):
    xpos = position_tsd['x']
    ypos = position_tsd['z']   
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    return occupancy


def computeAngularVelocity(angle, ep, bin_size = 400000):
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))    
    tmp2             = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)        
    time_bins        = np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
    index             = np.digitize(tmp2.index.values, time_bins)
    tmp3             = tmp2.groupby(index).mean()
    tmp3.index         = time_bins[np.unique(index)-1]+(bin_size/2)
    tmp3             = nts.Tsd(tmp3)
    tmp4            = np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
    velocity         = nts.Tsd(t=tmp3.index.values[1:], d = tmp4)
    velocity         = velocity.restrict(ep)
    return velocity

def computeAngularVelocityTuningCurves(spikes,hds, angle, ep, nb_bins = 20, bin_size = 100000):
    tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))    
    tmp2             = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)        
    time_bins        = np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
    index             = np.digitize(tmp2.index.values, time_bins)
    tmp3             = tmp2.groupby(index).mean()
    tmp3.index         = time_bins[np.unique(index)-1]+(bin_size/2)
    tmp3             = nts.Tsd(tmp3)
    tmp4            = np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
    velocity         = nts.Tsd(t=tmp3.index.values[1:], d = tmp4)
    velocity         = velocity.restrict(ep)
    bins             = np.linspace(-3*np.pi/2, 3*np.pi/2, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    velo_curves        = pd.DataFrame(index = idx, columns = hds)
    for k in hds:
        spks         = spikes[k]
        spks         = spks.restrict(ep)
        speed_spike = velocity.realign(spks)
        spike_count, bin_edges = np.histogram(speed_spike, bins)
        occupancy, _ = np.histogram(velocity, bins)
        spike_count = spike_count/(occupancy+1)
        velo_curves[k] = spike_count*(1/(bin_size*1e-6))

    return velo_curves

#plot(vel.loc[-np.pi:np.pi])

def slidingWinEp(ep,duration):
    t = np.arange(ep['start'].loc[0], ep['end'].loc[0], duration) #2mins
    t2 = np.hstack((t, ep['end'].loc[0]))
    t3 = np.repeat(t2,2,axis=0)
    t4 = t3[1:-1]
    t5 =t4.reshape(len(t4)//2,2)
    sw_ep=nts.IntervalSet(start=t5[:,0], end =t5[:,1])
    return sw_ep

def PFD_sw(ep,spikes,position,dur, hds): #duration must be in microsecs
    '''hds: list of hd cells index'''
    sw_ep=slidingWinEp(ep,dur)  
    max_tcurves=pd.DataFrame(index=range(len(sw_ep)), columns=hds)
    max_pRate=pd.DataFrame(index=range(len(sw_ep)), columns=hds)
    
    for i in range(len(sw_ep)):
        sw=sw_ep.loc[i]
        sw=nts.IntervalSet(sw.start,sw.end)
        tcurve=computeAngularTuningCurves(spikes,position['ry'],sw,60)
        for x,k in enumerate(hds):
            pFD=tcurve[k].idxmax(axis=0)
            pFR=tcurve[k].max()
            max_pRate.loc[i,k]=pFR
            max_tcurves.loc[i,k]=pFD

    return max_tcurves, max_pRate



def circStats_sw(ep,spikes,position,dur,hds):
    sw_ep=slidingWinEp(ep,dur)  
    c_mean=pd.DataFrame(index=range(len(sw_ep)), columns=hds)  
    c_var=pd.DataFrame(index=range(len(sw_ep)), columns=hds)  
    
    for i in range(len(sw_ep)):
        sw=sw_ep.loc[i]
        sw=nts.IntervalSet(sw.start,sw.end)
        for x,k in enumerate(hds):
            c_mean.loc[i,k]=circmean(position['ry'].realign(spikes[k].restrict(sw)))
            c_var.loc[i,k]=circvar(position['ry'].realign(spikes[k].restrict(sw)))

    return c_mean,c_var


def PFD_Rates_trim(ep,spikes,position,dur): #duration must be in microsecs
    sw_ep=slidingWinEp(ep,dur)
       
    max_tcurves=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    max_pRate=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    for i in range(len(sw_ep)):
        sw=sw_ep.loc[i]
        sw=nts.IntervalSet(sw.start,sw.end)
        tcurve=computeFrateAng(spikes,position,sw,60) #counts not normalised
        max_pRate.loc[i]=tcurve.max()
        max_tcurves.loc[i]=tcurve.idxmax(axis=0)
    return max_tcurves, max_pRate   




def PFD_Rates(ep,spikes,position,dur): #duration must be in microsecs
    '''computes the firing rates and corresponding angle'''
    sw_ep=slidingWinEp(ep,dur)       
    max_tcurves=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    max_pRate=pd.DataFrame(index=range(len(sw_ep)), columns=spikes.keys())
    for i in range(len(sw_ep)):
        sw=sw_ep.loc[i]
        sw=nts.IntervalSet(sw.start,sw.end)
        tcurve=computeFrateAng(spikes,position,sw,60)
        for k in spikes.keys():
            pFD=tcurve[k].idxmax(axis=0)
            pFR=tcurve[k].max()
            max_pRate.iloc[i,k]=pFR
            max_tcurves.iloc[i,k]=pFD
    return max_tcurves, max_pRate

def smallestSignedAngleBetween(x, y):
    "takes an array of x and y and returns the smallest diff in ang"
    dat=pd.DataFrame(index=np.arange(len(x)),columns=['delta'])
    for i in range(len(x)):
        a=(x[i]-y[i])%np.pi
        b=(y[i]-x[i])%np.pi
        if a<b:
            dat.iloc[i,0]=-a
        else:
            dat.iloc[i,0]=b
    return dat
            
def largestSignedAngleBetween(x, y):
    "takes an array of x and y and returns the smallest diff in ang"
    dat=pd.DataFrame(index=np.arange(len(x)),columns=['delta'])
    for i in range(len(x)):
        a=(x[i]-y[i])%np.pi*2
        b=(y[i]-x[i])%np.pi*2
        if a<b:
            dat.iloc[i,0]=-a
        else:
            dat.iloc[i,0]=b        
    
    # a = (x - y) % np.pi #if you want the largest difference, set it to modulo 2*np.pi for this and next line
    # b = (y - x) %  np.pi
    # -a if a < b else b
    return dat


def PFD(tcurve, ep,spikes): #duration must be in microsecs
#mean firing rate is the total number of emitted spikes divided by the total duration, ignoring occupancy
    max_tcurves=pd.DataFrame(index=tcurve.columns, columns=['PFD','peak_fr','mean_fr'])
    for k in tcurve.columns:
        pFD=tcurve[k].idxmax(axis=0)
        pFR=tcurve[k].max()
       # mFR=sum(tcurve[k].values)/len(tcurve[k])
        mFR=len(spikes[k].restrict(ep))/ep.tot_length('s')
        max_tcurves.loc[k,'PFD']=pFD
        max_tcurves.loc[k,'peak_fr']=pFR
        max_tcurves.loc[k,'mean_fr']=mFR 
        
    return  max_tcurves

  
    
    
def pcorrcoef(x, y, deg=False, test=True):
    '''Circular correlation coefficient of two angle data(default to degree)
    Set `test=True` to perform a significance test.
    '''
    convert = np.pi / 180.0 if deg else 1
    sx = np.frompyfunc(np.sin, 1, 1)((x - mean(x)) * convert)
    sy = np.frompyfunc(np.sin, 1, 1)((y - mean(y)) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())

    if test:
        l20, l02, l22 = (sx ** 2).sum(),(sy ** 2).sum(), ((sx ** 2) * (sy ** 2)).sum()
        test_stat = r * np.sqrt(l20 * l02 / l22)
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(test_stat)))
      
    return r,p_value 


      
def circular_stats(ep,spikes,position):
    circ_stats=pd.DataFrame(index=spikes.keys(), columns=['circ_mean','circ_var'])
    for i in spikes.keys():
        circ_stats.loc[i,'circ_mean']=circmean(position['ry'].realign(spikes[i].restrict(ep)))
        circ_stats.loc[i,'circ_var']=circvar(position['ry'].realign(spikes[i].restrict(ep)))
    return circ_stats
#astropy.stats.circstats.vonmisesmle--modify to include kappa!

def computeCircularStats(epochs,spikes,position, names,hds):
    circ_mean=pd.DataFrame(index=hds, columns=names)
    circ_var=pd.DataFrame(index=hds, columns=names)
    for n,ep in zip (names,epochs):
        for i,k in enumerate(hds):   
            circ_mean.loc[k,n]=circmean(position.realign(spikes[k].restrict(ep)))
            circ_var.loc[k,n]=circvar(position.realign(spikes[k].restrict(ep)))
    return circ_mean,circ_var




def shuffleByIntervalSpikes(spikes, epochs):
	shuffled = {}
	for n in spikes.keys():
		isi = []
		for i in epochs.index:
			spk = spikes[n].restrict(epochs.loc[[i]])
			tmp = np.diff(spk.index.values)
			np.random.shuffle(tmp)
			isi.append(tmp)
		shuffled[n] = nts.Ts(t = np.cumsum(np.hstack(isi)) + epochs.loc[0,'start'])
	return shuffled





#Stability
def stability(ep,spikes,pos):
    #dur= diff(ep)/2
     #duration must be in microsecs
    #ep=slidingWinEp(ep,dur)  
    '''TO DO
    modify the line above to accept arrays with multiple eps'''
    r=pd.DataFrame(index=spikes.keys(), columns=['spatial_corr','pval_corr'])
    ep1_start=ep.start[0]; ep1_end=ep.start[0]+(diff(ep)[0][0]/2)
            
    ep2_start=ep1_end; ep2_end=ep.end[0]
    
    ep1=nts.IntervalSet(ep1_start,ep1_end)
    ep2=nts.IntervalSet(ep2_start,ep2_end)
    tcurve1=computeAngularTuningCurves(spikes,pos,ep1,60)
    tcurve2=computeAngularTuningCurves(spikes,pos,ep2,60)
    
    for k in spikes.keys():
        r.loc[k,'spatial_corr']=astropy.stats.circcorrcoef(tcurve1[k].replace(nan,0).values,tcurve2[k].replace(nan,0).values)
        #r.loc[k,'spatial_corr']=scipy.stats.pearsonr(tcurve1[k].replace(nan,0).values,tcurve2[k].replace(nan,0).values)[0]
        #r.loc[k,'pval_corr']=scipy.stats.pearsonr(tcurve1[k].replace(nan,0).values,tcurve2[k].replace(nan,0).values)[1]
    return r
    


def computeStability_KA(epochs,spikes,position,name,hds): #computes spatial corr for several datafiles
    r=pd.DataFrame(index=np.arange(len(hds)), columns=name)
    for n, ep in zip(name,epochs): 
        #duration must be in microsecs  
        ep1_start=ep.start[0]; ep1_end=ep.start[0]+(diff(ep)[0][0]/2)
        
        ep2_start=ep1_end; ep2_end=ep.end[0]
        
        ep1=nts.IntervalSet(ep1_start,ep1_end)
        ep2=nts.IntervalSet(ep2_start,ep2_end)
    
        tcurve1=computeAngularTuningCurves(spikes,position,ep1,60)
        tcurve2=computeAngularTuningCurves(spikes,position,ep2,60)
        for i,k in enumerate(hds):
            tc=pd.concat((tcurve1[k],tcurve2[k]),axis=1)
            tc = tc.dropna(how = 'all')            
            tc=tc[(tc.iloc[:,0]>0.0) & (tc.iloc[:,1]>=0.0)]

            if tc.isnull().values.any():
                pass
            else:
                r.loc[i,n]=scipy.stats.pearsonr(tc.iloc[:,0],tc.iloc[:,1])[0]
    return r




def sessionStability(epochs,spikes,position,name,hds): #computes spatial corr for several datafiles
    r=pd.DataFrame(index=[0], columns=name)
    if len(hds)<2:
        r.iloc[0,0]=np.nan
        
    else:
    
        for n, ep in zip(name,epochs): 
            dur= diff(ep)/2
             #duration must be in microsecs  
            ep1_start=ep.start[0]; ep1_end=ep.start[0]+(diff(ep)[0][0]/2)
            
            ep2_start=ep1_end; ep2_end=ep.end[0]
            
            ep1=nts.IntervalSet(ep1_start,ep1_end)
            ep2=nts.IntervalSet(ep2_start,ep2_end)
        
            tcurve1=computeAngularTuningCurves(spikes,position,ep1,60)
            tcurve2=computeAngularTuningCurves(spikes,position,ep2,60)
           
            cCorrs=pd.DataFrame(index=np.arange(len(hds)),columns=['1','2'])
            for i,k in enumerate(hds):
                tc=pd.DataFrame([tcurve1[k],tcurve2[k]]).T
                tc=tc.dropna()
                tc1_mean=scipy.stats.circmean(tc.iloc[:,0])
                tc2_mean=scipy.stats.circmean(tc.iloc[:,1])
                
                # tc1_mean=scipy.stats.circmean(tcurve1[k].dropna())
                # tc2_mean=scipy.stats.circmean(tcurve2[k].dropna())
                cCorrs.iloc[i,:]=[tc1_mean,tc2_mean]
            
            cCorrs=cCorrs.dropna()
            cCorrs=np.unwrap(cCorrs)
                
            r.loc[0,n]=scipy.stats.pearsonr(cCorrs[:,0],cCorrs[:,1])[0] 
    return r


        
def corr_matrix(ep,spikes,position,hds):
    """
    sort cells according to their pref. fir direction first
    """
    dur= diff(ep)/2
     #duration must be in microsecs
    ep=slidingWinEp(ep,dur)  
     
    ep1_start=ep.start[0]; ep1_end=ep.start[0]+(diff(ep)[0][0]/2)
    
    ep2_start=ep1_end; ep2_end=ep.end[0]
    
    ep1=nts.IntervalSet(ep1_start,ep1_end)
    ep2=nts.IntervalSet(ep2_start,ep2_end)
    
    tcurve1=computeAngularTuningCurves(spikes,position,ep1,60)
    pfds=[scipy.stats.circmean(tcurve1[k]) for k in hds] 
    pfds_pair=pd.DataFrame(array(hds),array(pfds))
    pfd_sort=pfds_pair.sort_index(0)
    pfd_sort=list(pfd_sort.values.flatten())

    
    tcurve2=computeAngularTuningCurves(spikes,position,ep2,60)
    
    corr_mat=pd.DataFrame(index=np.arange(len(hds)),columns=np.arange(len(hds)))

    for v,j in enumerate(pfd_sort):
        for vv, k in enumerate(pfd_sort):
            corr_mat.loc[v,vv]=scipy.stats.pearsonr(tcurve1[j].values,tcurve2[k].values)[0]              
    return corr_mat.astype('float')






def centerTuningCurves(tcurve):
	"""
	center tuning curves by peak
	"""
	peak 			= pd.Series(index=tcurve.columns,data = np.array([circmean2(tcurve.index.values, tcurve[i].values) for i in tcurve.columns]));peak=peak.replace(NaN,0)
	new_tcurve 		= []
	for p in tcurve.columns:	
		x = tcurve[p].index.values - tcurve[p].index[tcurve[p].index.get_loc(peak[p], method='nearest')]
		x[x<-np.pi] += 2*np.pi
		x[x>np.pi] -= 2*np.pi
		tmp = pd.Series(index = x, data = tcurve[p].values).sort_index()
		new_tcurve.append(tmp.values)
	new_tcurve = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(new_tcurve).T, columns = tcurve.columns)
	return new_tcurve


def computeWidth(epochs,spikes, names,tcurves,hds):
    tc_width=pd.DataFrame(index=np.arange(len(hds)),columns=names)
    for ep, n, tc in zip(epochs, names,tcurves): 
        cen_tc=centerTuningCurves(tc)
        for i,k in enumerate(hds):
            try:
                tcurve=cen_tc[k]
                max_fr=tcurve.max(axis=0)
                min_fr=tcurve.min(axis=0)
                tc_half_w=(max_fr-min_fr)/2+min_fr
            
                tc_max_ang=tcurve.idxmax(axis=0)
                
                ls_tc=tcurve[tcurve.index < tc_max_ang]
                ls_fxn=scipy.interpolate.interp1d(ls_tc.values,ls_tc.index ,assume_sorted = False)
                ls=ls_fxn(tc_half_w)
        
                rs_tc=tcurve[tcurve.index > tc_max_ang]
                rs_fxn=scipy.interpolate.interp1d(rs_tc.values,rs_tc.index ,assume_sorted = False)
                rs=rs_fxn(tc_half_w)
                
                width=abs(ls)+rs
                tc_width.loc[i,n]=width
            except:
                tc_width.loc[i,n]=NaN
    return tc_width




def tc_width(tcurves,spikes):    
    'computes the width of the tuning curve of all cells'
    
    tc_width=pd.DataFrame(index=([0]),columns=spikes.keys())
    
    for i in tcurves:
        try:
            curves=centerTuningCurves(tcurves)
            tcurve=tcurves[i]
            max_fr=tcurve.max(axis=0)
            min_fr=tcurve.min(axis=0)
            tc_half_w=(max_fr - min_fr)/2 + min_fr
        
            tc_max_ang=tcurve.idxmax(axis=0)
            
            ls_tc=tcurve[tcurve.index < tc_max_ang]
            ls_fxn=scipy.interpolate.interp1d(ls_tc.values,ls_tc.index ,assume_sorted = False)
            ls=ls_fxn(tc_half_w)
    
            rs_tc=tcurve[tcurve.index > tc_max_ang]
            rs_fxn=scipy.interpolate.interp1d(rs_tc.values,rs_tc.index ,assume_sorted = False)
            rs=rs_fxn(tc_half_w)
            
            width=abs(ls)+rs
            
            tc_width.loc[0,i]=width
        except:
            tc_width.loc[0,i]=NaN
    return tc_width.T


def full_ang(ep,position):
    'generates a tsd with all the instances where all angular bins were sampled.' 
    starts=[]
    ends=[]
    count=np.zeros(60-1)
    bins=linspace(0,2*np.pi,60)
    ang_pos=position.restrict(ep)
    ang_time=ang_pos.times()
    
    idx=np.digitize(ang_pos,bins)-1
    
    start=0
    for i,j in enumerate(idx):
        count[j]+=1
        if np.all(count>=1):
            starts.append(start)
            ends.append(i)
            count=np.zeros(60-1)
            start=i+1
    
    t_start=ang_time[starts]
    t_end=ang_time[ends]
    full_ang_ep=nts.IntervalSet(start=t_start,end=t_end)
    
    return full_ang_ep





def full_ang2(ep,start_ep,position):
    '''generates a tsd with a time window where all angular bins were sampled
     inputs: ep- full session epoch
             start_ep- time to start the sampling win''' 

    count=np.zeros(60-1)
    bins=np.linspace(0,2*np.pi,60)
    
    new_ep=nts.IntervalSet(start=start_ep,end=ep.end)
    
    ang_pos=position.restrict(new_ep)
    ang_time=ang_pos.times()
    
    idx=np.digitize(ang_pos,bins)-1
    
    start=0
    for i,j in enumerate(idx):
        count[j]+=1
        if np.all(count>=1):
            t_start=ang_time[start]
            t_end=ang_time[i]
            full_ang_ep=nts.IntervalSet(start=t_start,end=t_end)
            break
        else:
            full_ang_ep=nts.IntervalSet(start=0,end=0)
    
    return full_ang_ep


def makeRingManifold(spikes, ep, angle, hds,neighbors=20,bin_size = 200):
    """
    spikes : dict of hd spikes
    ep : epoch to restrict
    angle : tsd of angular direction
    bin_size : in ms
    """
    neurons = hds#spikes.keys()
    inputs = []
    angles = []
    sizes = []
    bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[0]+bin_size, bin_size)
    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
        
    for i in neurons:
        spks = spikes[i].as_units('ms').index.values
        spike_counts[i], _ = np.histogram(spks, bins)
    
    rates = np.sqrt(spike_counts/(bin_size))
        
    epi = nts.IntervalSet(ep.loc[0,'start'], ep.loc[0,'end'])
    angle2 = angle.restrict(epi).as_units('ms')
    newangle = pd.Series(index = np.arange(len(bins)-1))
    tmp = angle2.groupby(np.digitize(angle2.index.values, bins)-1).mean()
    tmp = tmp.loc[np.arange(len(bins)-1)]
    newangle.loc[tmp.index] = tmp
    newangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

    tmp = rates.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values
    sizes.append(len(tmp))
    inputs.append(tmp)
    angles.append(newangle)

    inputs = np.vstack(inputs)

    imap = Isomap(n_neighbors = neighbors, n_components = 2, n_jobs = -1).fit_transform(inputs)    

    H = newangle.values/(2*np.pi)
    # HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    # RGB = hsv_to_rgb(HSV)
        
    # fig,ax=subplots(figsize=(5.1,4.7))
    # ax = subplot(111)
    # ax.set_aspect(aspect=1)
    # ax.scatter(imap[:,0], imap[:,1], c = RGB, marker = 'o', alpha = 0.5, zorder = 2, linewidth = 0, s= 40)    
      
    # '''# hsv
    # display_axes = fig.add_axes([0.2,0.25,0.05,0.1], projection='polar')
    # colormap = plt.get_cmap('hsv')
    # norm = mpl.colors.Normalize(0.0, 2*np.pi)
    # xval = np.arange(0, 2*pi, 0.01)
    # yval = np.ones_like(xval)
    # display_axes.scatter(xval, yval, c=xval, s=20, cmap=colormap, norm=norm, linewidths=0, alpha = 0.8)
    # display_axes.set_yticks([])
    # display_axes.set_xticks(np.arange(0, 2*np.pi, np.pi/2))
    # display_axes.grid(False)
    # '''
    # show()
    
    # ax.axis('on')

    
    return imap, H,bins[0:-1]+np.diff(bins)/2
    
def processed_files(animal_id,date):
    'input to the function must be a strings'
    
    data_dir='D:/EphysData/Experiments/'+ date+'/'+animal_id+'-'+date+'/'+animal_id+'-'+date+'/Analysis'
    dir='D:/EphysData/Experiments/'+ date+'/'+animal_id+'-'+date+'/'+animal_id+'-'+date
    epochs=nts.IntervalSet(pd.read_hdf(data_dir+'/BehavEpochs.H5'))
    position=pd.read_hdf(data_dir+'/'+'Position.H5')
    position = nts.TsdFrame(t = position.index.values, d = position.values, columns = position.columns, time_units = 's')
    spikes,shank= loadSpikeData(dir) #shank tells the number of cells on each shank

    tcurv={}
    for i in range(len(epochs)):
        tcurv[i]=computeAngularTuningCurves(spikes,position ['ry'],nts.IntervalSet(epochs.loc[i,'start'],epochs.loc[i,'end']),60)
    'tuning curves are computed based on entire epoch, to restrict it, just modify the end time above'
    #load tuning curve
    #np.load(dir+'/'+animal_id+'.npy').item()
    
    np.save(os.path.join(data_dir, animal_id), tcurv)  
    return spikes, epochs, position, tcurv   

def explore(eps, position):
    '''The function takes nts.TsdFrame of position from neuroseries and outputs total distance 
    traveled,distribution of distance traveled per frame and speed''' 
    
    expl=pd.DataFrame(index=range(len(eps)),columns=('tot_dist','speed'))
    for i in range(len(eps)):
        ep=nts.IntervalSet(start=eps.iloc[i,0],end=eps.iloc[i,1])
    
        pos_x=position['x'].restrict(ep)
        pos_y=position['z'].restrict(ep)
        
        x=array(pd.DataFrame(pos_x.values))
        y=array(pd.DataFrame(pos_y.values))
        
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]
        step_size = np.sqrt(dx**2+dy**2)#this must be used for computing the speed
        #step_size=np.concatenate(([[0]],step_size)) #new modification
        #dist = np.concatenate(([0], np.cumsum(step_size)))
        dist=sum(step_size)
        tot_dist=dist*100
        #tot_dist=dist[-1]*100 #converts to cm
        motion_frames=len(step_size[step_size>0])/120 #index only frames that the animal actually moved in secs
        speed=tot_dist/motion_frames        
        expl.iloc[i,0]=tot_dist
        expl.iloc[i,1]=speed
    return expl,step_size
'''
def explore_Center_Perimeter(eps,position, perArea=0.33):
    
    center_peri=pd.DataFrame(index=range(len(eps)),columns=('tot_dist','avg_speed','center_dist', 'perimeter_dist', 'center_speed','perimeter_speed'))
    for i in range(len(eps)):
        ep=nts.IntervalSet(start=eps.iloc[i,0],end=eps.iloc[i,1])
    
        pos=pd.DataFrame(index=(range(len(position.restrict(ep)))),columns=['x','z'])
        pos['x']=position['x'].restrict(ep).values
        pos['z']=position['z'].restrict(ep).values
        
        x_cen=(pos['x'].max()+pos['x'].min())/2
        y_cen=(pos['z'].max()+pos['z'].min())/2
        center=[x_cen,y_cen]
        
        tot_exp,dist=explore(ep,position)
        center_peri.loc[i,('tot_dist','avg_speed')]=tot_exp.values
        
        r=np.sqrt((pos['x']-x_cen)**2+(pos['z']-y_cen)**2) #len of the radius at all points
        cyl_r= r.max() #the radius of the area explored                   meters 56.2cm--cylinder size
        cyl_c=cyl_r-(perArea*cyl_r) #1/2of the cylinder 10cm from per
        
        #Center
        cen=dist[r[0:-1]< cyl_c]
       
        dist_c=sum(cen)*100
        dist_c_1=sum(cen) #bcos dis has already been converted to cms in the explore fxn
        vels_all_c=(cen*100)*120   #120 is the cam sampling freq, 100 brings the units to cm ####MUST DIVIDE NOT MULTIPLY!
        vel_c=dist_c/(len(cen)/120)
        
        #Wall
        wall=dist[r[0:-1]>=cyl_c]
        dist_w=sum(wall)*100
        vels_all_w=(wall*100)*120  
        vel_w=dist_w/(len(wall)/120)
    
        #Distance
        center_peri.loc[i,'center_dist']= dist_c
        center_peri.loc[i,'perimeter_dist']= dist_w        
      
        #Velocity
        center_peri.loc[i,'center_speed']= vel_c
        center_peri.loc[i,'perimeter_speed']= vel_w      
    return center_peri
'''
def explore_Center_Perimeter(eps,position,mouse_id,perArea=0.33):
    
    center_peri=pd.DataFrame(index=[mouse_id],columns=('tot_dist (m)','avg_speed (m/s)','center_dist (m)'\
                                                            ,'perimeter_dist (m)', 'center_speed (m/s)','perimeter_speed (m/s)'))
        
    dist_from_cen={}
    
    
    for i in range(len(eps)):
        ep=nts.IntervalSet(start=eps.iloc[i,0],end=eps.iloc[i,1])
    
        pos=pd.DataFrame(index=(range(len(position.restrict(ep)))),columns=['x','z'])
        
        #converts to cm
        x=array(position['x'].restrict(ep).values)
        y=array(position['z'].restrict(ep).values)
        
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]
        
        #convert step size to cm
        step_size = np.sqrt(dx**2+dy**2) #this must be used for computing the speed

        tot_dist=sum(step_size)
        
        motion_frames=len(step_size[step_size>0])/120 #index only frames that the animal actually moved in secs
        avg_speed=tot_dist/motion_frames 


        #Center
        x_cen=(x.max()+x.min())/2
        y_cen=(y.max()+y.min())/2
        center=[x_cen,y_cen]
        
        #tot_exp,dist=explore(ep,position)
        center_peri.loc[i,('tot_dist (m)','avg_speed (m/s)')]=[tot_dist,avg_speed]
        
        r=np.sqrt((x-x_cen)**2+(y-y_cen)**2) #len of the radius at all points
        dist_from_cen[mouse_id]=r
        
        
        
        cyl_r= r.max() #the radius of the area explored                   meters 56.2cm--cylinder size
        cyl_c=cyl_r-(perArea*cyl_r) #1/2of the cylinder 10cm from per
        
        #Center
        cen=step_size[r[0:-1]< cyl_c]
        cen_motion_frames=len(cen[cen>0])/120
        dist_c=sum(cen)
        #vels_all_c=cen*120   #120 is the cam sampling freq, 100 brings the units to cm ####MUST DIVIDE NOT MULTIPLY!
        vel_c=dist_c/cen_motion_frames
        
        #Wall
        wall=step_size[r[0:-1]>=cyl_c]
        wall_motion_frames=len(wall[wall>0])/120
        dist_w=sum(wall)
        #vels_all_w=(wall*100)*120  
        vel_w=dist_w/wall_motion_frames

        #Distance
        center_peri.loc[i,'center_dist (m)']= dist_c
        center_peri.loc[i,'perimeter_dist (m)']= dist_w        
      
        #Speed
        center_peri.loc[i,'center_speed (m/s)']= vel_c
        center_peri.loc[i,'perimeter_speed (m/s)']= vel_w      
    return center_peri,dist_from_cen



def all_frate_maps(spikes,position,ep,hds):
    
    tms={}
    GF, ext = computePlaceFields(spikes, position[['x', 'z']], ep, 50)
    for i,k in enumerate(hds):
       tms[i] = gaussian_filter(GF[k].values,sigma = 2)
    return tms
#Sanity Checks
#scatter(x[r<cyl_c],y[r<cyl_c],color='r',s=2,zorder=4)
#scatter(x[r>cyl_c],y[r>cyl_c])
#plot(p)

##################################
##MACHINE LEARNING FUNCTIONS
#################################
def randforestAccuracy(accuracy_mat):
    '''computes the percent accuracy based on a pandas crosstab mat of predictions'''
    val=[]

    if isinstance(accuracy_mat, pd.DataFrame)!=True:
        accuracy_mat=pd.DataFrame(accuracy_mat)
    for i in range(len(accuracy_mat)):
        r_pred=accuracy_mat.iloc[i,i]
        val.append(r_pred)    
    return  float(('%.1f' % (sum(val)/accuracy_mat.values.sum()*100)))

#####################################
# FIGURE FUNCTIONS
#####################################
def occu_heatmp(ep,position,_bins=50, threshold=0.13):   
    occu=computeOccupancy(position.restrict(ep),_bins)
    occu=gaussian_filter(occu,sigma=0.7)
    for i,z in enumerate(occu):
        for x,y in enumerate(occu):
            if occu[i][x] <=threshold:
                occu[i][x]=NaN
    fig, ax = plt.subplots()
    q=ax.imshow(occu,cmap='jet',interpolation = 'bilinear')
    cbar=fig.colorbar(q,orientation='vertical')
    cticks=cbar.ax.get_xticks()
    cbar.set_ticks([])
    #cbar.set_ticklabels(['min','max'])
    #cbar.ax.set_xlabel('occu')
    ax.axis('off')
    #make_axes(parents, location=None, orientation=None, fraction=0.15,)       
    #c=fig.colorbar(q, ticks=[occu.min(), occu.max()])#cax = fig.add_axes([0.78, 0.5, 0.03, 0.38]))
    #c.ax.set_xticklabels(['Low', 'High'])   
    plt.gca().invert_yaxis()
    return occu,fig

def path_spk_plt(ep,spikes,position):
    
    sz=len(spikes.keys())/3
    fig = figure(figsize = (15,16))
    fig.suptitle('Spikes + Path Plot',size=30)
    for i in spikes:
        ax=subplot(10,3,i+1)
        scatter(position['x'].realign(spikes[i].restrict(ep)),position['z'].realign(spikes[i].restrict(ep)),s=5,c='magenta',label=str(i))
        legend()
        plot(position['x'].restrict(ep),position['z'].restrict(ep),color='darkgrey', alpha=0.5)  
    return fig,ax

#ANIMATE TRAJECTORY IN XY


'''
px=position['x'].restrict(ep).as_units('s')
py=position['z'].restrict(ep).as_units('s')

nbins=np.linspace(px.index[0],px.index[-1],12000)
dig=np.digitize(px.index.values,nbins)

px=pd.DataFrame(px.values).groupby(dig).mean()
py=pd.DataFrame(py.values).groupby(dig).mean()


x = px.values.flatten()
y = py.values.flatten()
fig, ax = plt.subplots()
line, = ax.plot(x, y, color='k',linewidth=2,alpha=0.5)

def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    #line.axes.axis([0, 10, 0, 1])
    return line

ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],
                              interval=0.0001, blit=False, repeat=False)
'''



def bkgrid ():
    plt.gcf()
    pad=0.05
    ax=gcf().add_axes([pad,pad,1-(2*pad),1-(2*pad)])
    ax.grid(True,which='both',zorder=0)
    fx=lambda x: round(x,2)
    x=np.linspace(pad,1-pad,16)
    fig_lim=list(map(fx,x))
    ax_lim=np.linspace(0,1,16)
    ax.set_xticks(ax_lim)
    ax.set_xticklabels(fig_lim,size=6.5,color='r')
    ax.set_yticks(ax_lim)
    ax.set_yticklabels(fig_lim,size=6.5,color='r')
    return ax




def path_plot(eps,position):
    #fig=figure(figsize=(15,16))
    fig=figure()
    for i in range(len(eps)):
        if len(eps)==1:
            ax=subplot()
        else:    
            ax=subplot(1,len(eps),i+1)
        ep=eps.iloc[i]
        ep=nts.IntervalSet(ep[0],ep[1])
        plot(position['x'].restrict(ep),position['z'].restrict(ep),color='red',label=str(i), alpha=0.5) 
        legend()

def remove_polarAx(ax, xtcklabels=False ):
    #ax.set_yticks([])
    xticks=ax.xaxis.get_major_ticks()
    tcks=[1,3,5,7]
    for tck in tcks:
        xticks[tck].set_visible(False)
    if xtcklabels==True:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(['0\xb0','','90\xb0','','180\xb0  ','','270\xb0'])
        
def remove_box(num=2):
    if num >2:
        gca().spines['right'].set_visible(False)
        gca().spines['left'].set_visible(False)
        gca().spines['top'].set_visible(False)
    else:
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)

    

    
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=False):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=3, align='edge', width=widths,
                     edgecolor='k', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches



def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)
        
def makeODTarena(l=1.3):

    gca().plot([17,17],[9,240],color='k',zorder=3,linewidth=l)
    gca().plot([238,238],[9,95],color='k',zorder=3,linewidth=l)
    gca().plot([238,238],[150,240],color='k',zorder=3,linewidth=l)

    gca().plot([17,238],[9,9],color='k',zorder=3,linewidth=l)
    gca().plot([17,238],[240,240],color='k',zorder=3,linewidth=l)


    gca().plot([270,270],[9,95],color='k',zorder=3,linewidth=l)
    gca().plot([270,270],[150,240],color='k',zorder=3,linewidth=l)

    gca().plot([485,485],[9,240],color='k',zorder=3,linewidth=l)
    gca().plot([270,485],[9,9],color='k',zorder=3,linewidth=l)
    gca().plot([270,485],[240,240],color='k',zorder=3,linewidth=l)

    gca().plot([238,270],[95,95],color='k',linewidth=l)
    gca().plot([238,270],[150,150],color='k',linewidth=l)


    avers=gca().scatter(298,120,s=100, facecolors='red',edgecolors='k',linewidth=1,zorder=3)
    avers1=gca().scatter(436,120,s=100, facecolors='red',edgecolors='k',linewidth=1,zorder=3)

    neut=gca().scatter(210,120,s=100, facecolors='white', edgecolors='k',linewidth=1,zorder=3 )
    neut1=gca().scatter(68,120,s=100, facecolors='white', edgecolors='k',linewidth=1,zorder=3 )
    gca().axis('off')
    return  neut,avers


def angularDiff(ang_pair,ang_dir):
    ang_diff=[]
    if ang_dir=='clock':
        ang_diff.append((ang_pair.iloc[:,0]-ang_pair.iloc[:,1]) % pi)
    elif ang_dir=='anti':
        ang_diff.append((ang_pair.iloc[:,1]-ang_pair.iloc[:,0]) % pi)
    return ang_diff


def predAng(circ_mean,cond3):  
    '''cond3 must be in degrees'''
    
    data=pd.DataFrame(index=np.arange(len(circ_mean)),columns=['obs','pred'])
         
    for i in range(len(circ_mean)): 
        a=circ_mean.iloc[i,0]

        b=circ_mean.iloc[i,1]

        distance = (b - a) % (2*pi)
        if distance < - np.pi:
            distance += (2*pi);
        elif distance > deg2rad(179):
            distance -= (2*pi);

        delta = abs(distance)- deg2rad(cond3)              
        
        observed=b
        predicted=b+abs(delta)  
        data.iloc[i,:]=(observed,predicted)

    return data




def fullAngDrift2(epoch,spikes,position,hds,eps):
    '''in degs'''
    pfd_delta=pd.DataFrame(index=np.arange(len(eps)),columns=hds)
    pfd_cm=pd.DataFrame(index=np.arange(len(eps)),columns=hds)
    
    for j in range(len(eps)):
        ep=nts.IntervalSet(start=eps.iloc[j,0], end=eps.iloc[j,1])
        tc=computeAngularTuningCurves(spikes,position['ry'],ep,60)
        pfd_delta.loc[j]=rad2deg(tc.idxmax())
    pfd1= np.unwrap(pfd_delta,axis=0,discont=360)
    
    pfd_smooth=pd.DataFrame([gaussian_filter(pfd1[:,i].astype('float'),sigma=1.5) for i in range(pfd1.shape[1])]).T
    pfd_smooth.columns=hds    
    return  pfd_smooth,eps




def computeAngularVelocity2(angle, time_bins, ep):
    angle      = angle.restrict(ep)
    tmp        = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))    
    tmp2       = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=30.0)
    tmp2       = nts.Tsd(tmp2)
    index      = np.digitize(tmp2.as_units('ms').index, time_bins)
    tmp3       = tmp2.as_series().groupby(index).mean().reset_index(drop=True)
    tmp4       = np.diff(tmp3.values)/np.diff(time_bins*1e-3)[0]# converts to seconds
    return tmp4        