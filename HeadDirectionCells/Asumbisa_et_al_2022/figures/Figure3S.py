# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:00:17 2022

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
# bkAx=bkgrid() #helper fxn to set grid in background


#STABILITY 1st vs 2nd
gs_dd2=fig.add_gridspec(2,1, left=0.58,right=0.7,bottom=0.55,top=0.9,hspace=0.4,wspace=1.3)
fig.add_subplot(gs_dd2[0])
data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_stabilityL2Eps.npy',allow_pickle=True)

x=data[2][0].T #2===gnat
for i in range(1,len(data[2])):
    x=pd.concat([x,data[2][i].T])

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


#STABILITY Exposure 1 vs 2
fig.add_subplot(gs_dd2[1])
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_Exposure1_2_3strains.npy',allow_pickle=True)

x1=np.unwrap(dat[2])
plt.scatter(x1[:,0],abs(x1[:,1]),c='gray',s=5,alpha=0.7,edgecolor='k')
gca().set_aspect('equal')
gca().set_ylim([0,8])
gca().set_xlim([0,8])
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


#DARK vs LIGHT - rd1

gs_dd3=fig.add_gridspec(3,2, left=0.77,right=0.88,bottom=0.55,top=0.9,hspace=0.4,wspace=1.3)

dat_dark1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_LightDark3_paired3Strains.npy',allow_pickle=True)  
gn_dark=dat_dark1[2]

fig.add_subplot(gs_dd3[0])
#Mean Firing Rate
expC='means'
x=gn_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=4)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_xticklabels([])


plot([0,1],[40,40],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.27,40.5),size=small)
gca().set_ylabel('Mean rate (Hz)',labelpad=-0.3)
remove_box()
tickParams()

fig.add_subplot(gs_dd3[1])
expC='peaks'
x=gn_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=4)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_xticklabels([])
gca().set_yticks([0,40,80])
gca().set_yticklabels([0,40,80])

plot([0,1],[95,95],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.27,96),size=small)
gca().set_ylabel('Peak rate (Hz)',labelpad=-0.3)
remove_box()
tickParams()


fig.add_subplot(gs_dd3[2])
expC='mvl'
x=gn_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=2)
gca().set_xticklabels([])

gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)

gca().set_yticks([0,0.5,1]);gca().set_yticklabels(['0','.5','1'],size=small)
gca().set_ylim([0,1.1])
plot([0,1],[0.99,0.99],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.25,1.007),size=small)
gca().set_ylabel('Vector length (r)',size=small,labelpad=2)
remove_box()
tickParams()


fig.add_subplot(gs_dd3[3])
expC='info'
x=gn_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=4)
gca().set_xticklabels([])
gca().set_xlim(-0.1,1.1)
gca().set_xticks([0,1])
plot([0,1],[2,2],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.25,2.05),size=small)
gca().set_ylabel('Info. (bits/spk)',labelpad=3)
gca().set_yticks([0,1,2]);gca().set_yticklabels(['0','1','2'])
remove_box()
tickParams()


fig.add_subplot(gs_dd3[4])
expC='width'
x=gn_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=4)
plot([0,1],[3.4,3.4],color='k',linewidth=1)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)

gca().set_xticklabels(['Light','Dark'])
xticks(rotation=30)
gca().set_yticks([deg2rad(45),deg2rad(90),deg2rad(135),deg2rad(180)]);
gca().set_yticklabels(['45\u00b0','90\u00b0','135\u00b0','180\u00b0'])
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.26,3.43),size=small)
gca().set_ylabel('Tuning width',labelpad=-2)
remove_box()
tickParams()
gca().tick_params(axis='y',which='both',pad=0.05, length=2)



#Bayesian Decoding
gs10=fig.add_gridspec(1,1, left=0.59,right=0.88,bottom=0.11,top=0.23)
fig.add_subplot(gs10[:])
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure1_decoding_exampleGN.npy',allow_pickle=True).item()
a1=dat['actual']
a2=dat['decode']

# #light decoding
plot(a1[140:740], label='Actual', color='royalblue',alpha=0.9,linewidth=1.5)
plot(a2[140:740],label='Decoded',color='darkorange',alpha=0.8,linewidth=1.5)
gca().spines['left'].set_position(('axes',-0.01))

gca().set_xticks(np.linspace(0,600,3))
gca().set_xlim([0,600])
gca().set_xticklabels(['0','1','2'])
gca().set_xlabel('Time (min)',labelpad=0)

gca().set_ylim([0,2*np.pi])
gca().set_yticks([0,2*np.pi])
gca().set_yticklabels(['0\u00b0','360\u00b0'])
gca().set_ylabel('Head direction',labelpad=-4)
legend(bbox_to_anchor=(1.02, 1.28), loc='upper right',ncol=2,frameon=True,columnspacing=1,handlelength=2,handletextpad=0.3)
remove_box()
tickParams()
gca().tick_params(axis='y',which='major',pad=0.3, length=2,tickdir='out')
title('Mean Absolute Error= 29.2\u00b0 (14 HD cells)',loc='left',size=xsmall)



#Visual Cue Control
gs11=fig.add_gridspec(2,2, left=0.59,right=0.71,bottom=0.34,top=0.46,hspace=0.6,wspace=0.6)
d=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_tcurve_cueRotLight_gn90deg.npy',allow_pickle=True).item()
tc1=d['tc_A']
tc2=d['tc_B']

ids=[0,6,5,2]
ct=0
for i,x in enumerate(ids):
    ax=fig.add_subplot(gs11[i])
    remove_box()
    st=plot(tc1[x],c='k')
    bl_n=plot(tc2[x],c='#DC3C95',linestyle=(0,(3,3)),label='Gnatmut: 90\u00b0'+'Visual cue rot.')
    gca().set_yticks([])
    gca().set_xticks([0,2*np.pi])
    gca().set_xticklabels([])
    gca().set_xlim([0,2*np.pi])
    tickParams()
    plt.annotate('C'+str(ct+1),(pi,round(tc2[x].max())+4),xycoords='data',size=xsmall)
    
    gca().set_yticks([0,round(tc2[x].max())+5])
    gca().set_yticklabels([0,round(tc2[x].max())+5])
    if i==0:
        gca().set_yticks([0,round(tc1[x].max())+5]) 
        gca().set_yticklabels([0,round(tc1[x].max())+5])
        legend(loc='upper right',bbox_to_anchor=(1.8,2),frameon=False,labelspacing=0.12,handlelength=2,handletextpad=0.3)

        
    if i==2:
        plt.ylabel('Firing rate (Hz)',y=1.2,labelpad=0)
    if i in [2,3]:

        
        gca().set_xticks([0, 2*pi])
        gca().set_xticklabels(['0\u00b0','360\u00b0'])
    if i==3:
    
        gca().set_xlabel('Head direction',x=-0.4,labelpad=0)

    ct+=1


gs12=fig.add_gridspec(1,1, left=0.76,right=0.88,bottom=0.34,top=0.42)
fig.add_subplot(gs12[0])

dat=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_cueRots3_Gain_ver3.npy',allow_pickle=True)

gn=dat[2]
bins=np.linspace(0,1.2,20)
hist(gn,bins, histtype='stepfilled',density=True,edgecolor='k', color='#DC3C95', alpha=0.85,linewidth=1,label='rd1')
remove_box()
gca().set_xticks([0,0.5,1])
xlim(0,1.1)
gca().set_xticklabels(['0',0.5,1])
gca().set_yticks([0,2,4])
gca().set_yticklabels(['0','2','4'])
gca().set_ylabel('Normalized counts',labelpad=-0.2)

tickParams()
gca().set_xlabel('Gain (Visual cue rotation)',labelpad=0)


# #Panel Labels
# plt.annotate('A',(0.06,0.94),xycoords='figure fraction',size=large, fontweight='bold')
# plt.annotate('B',(0.06,0.43),xycoords='figure fraction',size=large, fontweight='bold')
# plt.annotate('C',(0.41,0.94),xycoords='figure fraction',size=large, fontweight='bold')
