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


def computeFiringRates(spikes, epochs, tcs,name,hds):
    mean_frate = pd.DataFrame(index = np.arange(len(hds)), columns = name)
    peak_frate= pd.DataFrame(index = np.arange(len(hds)), columns = name)
    pfd= pd.DataFrame(index = np.arange(len(hds)), columns = name)
    
    for n, ep, tc in zip(name, epochs,tcs):
        for i,k in enumerate(hds):
            mean_frate.loc[i,n] = len(spikes[k].restrict(ep))/ep.tot_length('s') 
            peak_frate.loc[i,n]=tc[k].max()
            pfd.loc[i,n]=tc[k].idxmax(axis=0)  
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
                   


def MutualInfo(spikes,ep,position,hds):
    I=pd.DataFrame(index=hds,columns=[0])
    for k in hds: 
        lamda_i=computeAngularTuningCurves(spikes,position,ep,60)[k].values
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


def PFD_Rates(ep,spikes,position,dur): #duration must be in microsecs
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


def makeRingManifold(spikes, ep, angle, hds,neighbors=50,bin_size = 200):
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
    
    return imap, H,bins[0:-1]+np.diff(bins)/2



def bkgrid ():
    '''Helper fxn for making grids in matplotlib during figure making'''
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


def makeBettiFromRing(spk_iso):
    barcodes = tda(spk_iso, maxdim=1, coeff=2)['dgms']
    h1 = barcodes[1]
    
    betti_x=h1[:,0] #birth 
    betti_x1=h1[:,1] #death
    
    r=max(abs(betti_x1-betti_x)) #the longest radius
    
    return betti_x, betti_x1, r



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