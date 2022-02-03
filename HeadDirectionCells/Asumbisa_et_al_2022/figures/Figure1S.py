# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:53:55 2022

@author: kasum
"""

import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers import *
from pylab import *
import os, sys
from functions import *
from pycircstat.tests import rayleigh

from astropy.visualization import hist
import statsmodels.api as sm
import scipy
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage,AnchoredOffsetbox
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
a=mpl.rcParams.keys() #to get the gloal params
from matplotlib.lines import Line2D

###############################################################################
#SETTING DEFAULTS FOR MATPLOTLIB
###############################################################################
mpl.rcParams['pdf.fonttype']= 42
mpl.rcParams['ps.fonttype']=42
mpl.rcParams['font.family']='Arial'

xlarge=14
large=12
med=10
small=8
xsmall=6

plt.rc('figure', titlesize=large)   # figure title
plt.rc('font', size=small)          # default text sizes
plt.rc('axes', titlesize=med)       # axes title
plt.rc('axes', labelsize=small)     # x,y labels
plt.rc('xtick', labelsize=small)    # xtick labels
plt.rc('ytick', labelsize=small)    # ytick labels
plt.rc('legend', fontsize=xsmall)   # legend


def tickParams(axis='both'):
    gca().tick_params(axis=axis,which='major',pad=1.5, length=2)
###############################################################################


fig=plt.figure(figsize=(11,8.5))
#bkAx=bkgrid() #helper fxn to set grid in background


##########################################################
# A. 1st vs 2nd Half mean PFD (WT-Light)
##########################################################

gs6=fig.add_gridspec(2,2, left=0.08,right=0.2,bottom=0.35,top=0.47,hspace=0.6,wspace=0.6)
#1D TCURVES
tc=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/PAPER1/data/figures/figure2_example_tcurves_wtL.npy',allow_pickle=True).item()
ep=tc['ep']
all_spks=tc['spk']
pos=tc['pos']
ids=[0,1,2,4]

eps=slidingWinEp(ep,3e+8)
tc1=computeAngularTuningCurves(all_spks,pos['ry'],nts.IntervalSet(eps.loc[0].start,eps.loc[0].end),60)[ids]
tc2=computeAngularTuningCurves(all_spks,pos['ry'],nts.IntervalSet(eps.loc[1].start,eps.loc[1].end),60)[ids]

ct=0
for i in ids:
    fig.add_subplot(gs6[ct])

    plt.annotate('C'+str(ct+1),(pi,round(tc2[i].max())+0.85),xycoords='data',size=xsmall)
    plot(tc1[i],color='gray', label='1st Half')
    plot(tc2[i],color='k', linestyle=(0,(3,3)), label='2nd Half')
    gca().set_xticks([0,2*np.pi])
    gca().set_xticklabels([])
    gca().set_xlim([0,2*np.pi])
    gca().set_yticks([0,round(tc2[i].max())+1])
    gca().set_yticklabels([0,round(tc2[i].max())+1])
    gca().set_aspect('auto')
    remove_box()
    tickParams()

    if ct==1:
        legend(bbox_to_anchor=(0.6, 1.7), loc='upper right',ncol=2,frameon=False,columnspacing=1,handlelength=2,handletextpad=0.2)
        #plt.text(-16,75,'1D tuning curves',size=12)
    if ct==2:
        plt.ylabel('Firing rate (Hz)',y=1.2,labelpad=0)


    if i in [2,4]:
        gca().set_xlabel('')
        gca().set_xticks([0, 2*pi])
        gca().set_xticklabels([0, 2*pi])
        gca().set_xticklabels(['0\u00b0','360\u00b0'])
    ct+=1
gca().set_xlabel('Head direction',x=-0.4,labelpad=0)




gs7=fig.add_gridspec(1,1, left=0.25,right=0.37,bottom=0.35,top=0.47)
gs_dd1=fig.add_subplot(gs7[0])
data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_stabilityL2Eps.npy',allow_pickle=True)

x=data[0][0].T
for i in range(1,len(data[0])):
    x=pd.concat([x,data[0][i].T])

x1=np.unwrap(x)
plt.scatter(x1[:,0],abs(x1[:,1]),c='gray',s=5,alpha=0.7,edgecolor='k')
gca().set_aspect('equal')
gca().set_ylim([0,8])
gca().set_xlim([0,8])
plt.plot([0, 1], [0, 1], transform=gca().transAxes,color='red', linestyle=(0,(3,3)),linewidth=1,zorder=3)
gca().set_yticks([0,2*pi])
gca().set_yticklabels(['0\u00b0','360\u00b0'])
gca().set_ylabel('Mean PFD (2nd Half)',labelpad=-0.3)
gca().set_xticks([0,2*pi])
gca().set_xticklabels(['0\u00b0','360\u00b0'])
gca().set_xlabel('Mean PFD (1st Half)',labelpad=0)
p=scipy.stats.pearsonr(x1[:,0].astype('float'),(x1[:,1]).astype('float'))[0]
plt.annotate('r = '+ str(round(p,2))+', p< 0.0001',(0.5,7.5),size=xsmall)
remove_box()
tickParams()


##########################################################
# B. Exposure 1 vs Exposure 2 mean PFD (WT-Light)
##########################################################

gs8=fig.add_gridspec(2,2, left=0.43,right=0.55,bottom=0.35,top=0.47,hspace=0.6,wspace=0.6)
tc_12=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_exposure12_tc_WT.npy',allow_pickle=True).item()
id_=[8,5,1,6]
tc1=tc_12['tc_A']
tc2=tc_12['tc_B']

ct=0
for i in id_:
    fig.add_subplot(gs8[ct])

    plot(tc1[i],color='gray', label='Exposure 1')
    plot(tc2[i],color='k', linestyle=(0,(3,3)), label='Exposure2')
    remove_box()

    gca().set_xticks([0,2*np.pi])
    gca().set_xticklabels([])
    gca().set_xlim([0,2*np.pi])
    gca().set_yticks([0,round(tc1[i].max())+1])
    gca().set_yticklabels([0,round(tc1[i].max())+1],size=small)

    plt.annotate('C'+str(ct+1),(pi,round(tc1[i].max())+0.85),xycoords='data',size=xsmall)
    tickParams()
    if ct==1:
        legend(bbox_to_anchor=(1, 1.7), loc='upper right',ncol=2,frameon=False,columnspacing=1,handlelength=2,handletextpad=0.2)
    if ct==2:
        plt.ylabel('Firing rate (Hz)',y=1.2,labelpad=0)
    if ct not in [1,3]:
        gca().set_yticks([0,round(tc2[i].max())+1])
        gca().set_yticklabels([0,round(tc1[i].max())+1])

    if ct in [2,3]:
        gca().set_xlabel('')
        gca().set_xticks([0, 2*pi])
        gca().set_xticklabels([0, 2*pi])
        gca().set_xticklabels(['0\u00b0','360\u00b0'])
    ct+=1
gca().set_xlabel('Head direction',x=-0.4,labelpad=0)



gs9=fig.add_gridspec(1,1, left=0.6,right=0.72,bottom=0.35,top=0.47)
fig.add_subplot(gs9[0])
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_Exposure1_2_3strains.npy',allow_pickle=True)

x1=np.unwrap(dat[0])
plt.scatter(x1[:,0],abs(x1[:,1]),c='gray',s=5,alpha=0.7,edgecolor='k')
gca().set_aspect('equal')
gca().set_ylim([0,8.5])
gca().set_xlim([0,8.5])
plt.plot([0, 1], [0, 1], transform=gca().transAxes,color='red', linestyle=(0,(3,3)),linewidth=1,zorder=3)
gca().set_yticks([0,2*pi])
gca().set_yticklabels(['0\u00b0','360\u00b0'])
gca().set_ylabel('Mean PFD (Exposure 2)',labelpad=0)
gca().set_xticks([0,2*pi])
gca().set_xticklabels(['0\u00b0','360\u00b0'])
gca().set_xlabel('Mean PFD (Exposure 1)',labelpad=0)

p=scipy.stats.pearsonr(x1[:,0].astype('float'),(x1[:,1]).astype('float'))[0]
plt.annotate('r = '+ str(round(p,2))+', p< 0.0001',(0.5,7.5),size=xsmall)
remove_box()
tickParams()


#Panel labels
plt.annotate('A',(0.025,0.51),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('B',(0.4,0.51),xycoords='figure fraction',size=large,fontweight='bold')

# fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Fig1_S1_4_py.pdf',dpi=600, format='pdf',transparent=True)



