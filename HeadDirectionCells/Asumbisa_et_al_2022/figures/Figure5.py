# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:52:50 2022

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
    gca().tick_params(axis=axis,which='major',pad=2, length=2)
###############################################################################


fig=plt.figure(figsize=(11,8.5))
# bkAx=bkgrid() #helper fxn to set grid in background

#################################################################################
#A. wtD vs blind OSN
#################################################################################

gsA=fig.add_gridspec(3,2,left=0.05,right=0.17,top=0.9,bottom=0.74,wspace=-0.1)

tc_osnB=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig3_OSN_tc_rd.npy',allow_pickle=True)[1]
ids=[1,3,6]
for i,x in enumerate(ids):
    fig.add_subplot(gsA[i,1],projection='polar')
    plot(tc_osnB['osn'][x],linewidth=1.5,c='#907398',zorder=6)
    gca().fill(tc_osnB['osn'][x],color='white',zorder=5)
    if i==0:
        title('Olfaction ablated\nBlind',size=small) 
    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    
    

tc_darkWT=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\Figure1\figure1_example_tcurves.npy',allow_pickle=True) .item()
ids=[22,13,10]
for i,x in enumerate(ids):
    fig.add_subplot(gsA[i,0],projection='polar')
    plot(tc_darkWT['dark'][x],linewidth=1.5,c='grey',zorder=6)
    gca().fill(tc_darkWT['dark'][x],color='white',zorder=5)
    if i==0:
        title(r'WT$_D$',size=small) 
    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
   



#################################################################################
#A1.) wtD vs blind OSN
#################################################################################
gsA=fig.add_gridspec(3,2,left=0.05,right=0.17,top=0.9,bottom=0.74)
gsA1=fig.add_gridspec(1,1,left=0.06,right=0.16,top=0.73,bottom=0.65)

data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_OSN_WTd_Blind_ver2.npy',allow_pickle=True)
ndat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_stnd_WTd_LD_session.npy',allow_pickle=True).item()


##############################################################################
###Mean Vector Length ##########################################################
con='vlength'
a2=ndat[con]
b= np.concatenate((data[1][con],data[2][con]))

ax1=fig.add_subplot(gsA1[0])

vp=ax1.violinplot([list(a2.values.flatten()),list(b.flatten())],showmeans=False, showmedians=False, showextrema=False,positions=[0.9,2.2])#widths=[0.14,0,0.82]
bp=boxplot([a2.values.flatten(),b.flatten()],positions=[0.9,2.2],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','#986D97']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
gca().set_yticks([0,0.5,1]); gca().set_yticklabels(['0','.5','1'])
gca().set_ylabel('Vector length (r)',labelpad=0)
gca().set_ylim(-0.02,1.02)
plot([0.9,2.2],[0.95,0.95],color='k',linewidth=1)
plt.annotate('**',(1.40,0.87),size=xlarge)
tickParams()
gca().set_xticklabels([])

gca().tick_params(axis='x',labelrotation=0,pad=3,length=2,direction='in')



#################################################################################
#B.) Floor control - Gain
#################################################################################
gsB=fig.add_gridspec(1,1,left=0.24,right= 0.41,top=0.83,bottom=0.69)
fig.add_subplot(gsB[0])

dat=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_flrRots3_Gain_ver3.npy',allow_pickle=True)

wt=dat[0]
blind=dat[1]+dat[2]
bins=np.linspace(0,1.2,20)
hist(blind,bins, histtype='stepfilled',density=True,edgecolor='k', color='#907398', alpha=0.85,linewidth=1,label='Blind')
hist(wt,bins, histtype='stepfilled',density=True, edgecolor='k',color='darkgray',linewidth=1,label=r'WT$_D$',alpha=0.9)
remove_box()

gca().set_xticks([0,0.5,1])
gca().set_xticklabels(['0',0.5,1])

gca().set_yticks([0,1.5,3])
gca().set_yticklabels(['0','1.5','3'])

legend(loc='upper right',bbox_to_anchor=(1.09,1.01),frameon=0,handlelength=2,handletextpad=0.3,ncol=1)
gca().set_ylabel('Normalized counts',labelpad=-0.2)

plt.annotate('p< 0.0001',(0.2,3.19),size=xsmall)
gca().set_xlabel('Gain (Floor control)',labelpad=0,x=0.42)
tickParams()


#####################################################################################
# C. ADN units characterised as HD cells
#####################################################################################

gsD=fig.add_gridspec(1,1,left=0.49,right=0.61 ,top=0.9,bottom=0.69)
ax2=fig.add_subplot(gsD[:])
#E Distribution of HD counts

countsOSN=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_hdsDistributionLightDark-OSN_4_ver2.npy',allow_pickle=True)
light_counts=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_hdsDistributionLight3_ver1.npy',allow_pickle=True)
dark_counts=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_hdsDistributionDark3_ver1.npy',allow_pickle=True)

a=light_counts[0]['hd_%']   #WT_light
a1=countsOSN[0]['hd_%']     #WT_light OSN
b=dark_counts[0]['hd_%']    #WT_dark
b1=countsOSN[3]['hd_%']     #WT_dark_OSN

vp=ax2.violinplot([list(b),list(b1)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([b,b1],positions=[0,1],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','darkgray']
alp=[0.9,0.5]
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(alp[i])
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


gca().set_xticklabels([r'WT$_D$',r'WT$_D$'+'\nOlfaction ablated'],size=small)
gca().set_ylim(-2,100)
gca().set_yticks([0,25,50,75,100])
gca().set_ylabel('ADn units characterised as\nHD cells (%)',labelpad=-2)
remove_box()
tickParams()

gsC2=fig.add_gridspec(1,1,left=0.48,right=0.59,top=0.95,bottom=0.9)
axC2=fig.add_subplot(gsC2[0])
plot([0.36,0.94],[0.1,0.1],linewidth=1,color='k')
plt.annotate('**',(0.6,0.027),size=xlarge)
gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC2.axis('off')




######################################################################################################
#D. WT post OSN Ablation
#####################################################################################################
gsC=fig.add_gridspec(3,2,left=0.65,right=0.77,top=0.9,bottom=0.74,wspace=-0.1)

rd_tc=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig3_OSN_tc_rd.npy',allow_pickle=True)
wt_tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_OSN_tc_wtLD.npy',allow_pickle=True)

tc1=wt_tc[0]['pre_tc']
tc2=wt_tc[0]['post_tc']

ids=[7,8,9]
#192---good tc for LD osn
for i,x in enumerate(ids):
    fig.add_subplot(gsC[i,0],projection='polar')
    plot(tc1[x],linewidth=1.5,c='#56B4E9',zorder=6)
    gca().fill(tc1[x],color='white',zorder=5)
    if i==0:
        title(r'WT$_L$',size=small)
    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    plt.annotate('C'+str(i+1),(-13,11),xycoords='axes points', size=small)

for i,x in enumerate(ids):
    fig.add_subplot(gsC[i,1],projection='polar')
    plot(tc2[x],linewidth=1.5,c='darkgray',zorder=6)
    gca().fill(tc2[x],color='white',zorder=5)
    if i==0:
        title(r'WT$_D$',size=small)
    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
title('Olfaction ablated',y=3.7,x=-0.43,size=small)



gsC1=fig.add_gridspec(1,1,left=0.68,right=0.74,top=0.73,bottom=0.65)
data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_OSN_wtLD_xtics.npy',allow_pickle=True).item()

con='mvl'
a=data[con].light.values.flatten()
b= data[con].dark.values.flatten()

fig.add_subplot(gsC1[0])

plt.plot([0,1],[a,b],color='gray', alpha=0.5, linewidth=0.8)
plt.plot([0,1],[median(a),median(b)],color='red',linewidth=1.5,zorder=2)
plt.scatter(np.zeros(len(a)),a,color='#56B4E9',s=5,zorder=3,alpha=0.5)
plt.scatter(np.ones(len(b)),b,color='k',s=5,zorder=3,alpha=0.5)
remove_box()
gca().set_yticks([0,0.5,1]); gca().set_yticklabels(['0','.5','1'])
gca().set_ylabel('Vector length (r)',labelpad=0)
gca().set_ylim(-0.02,1.02)
tickParams()
plt.tick_params(axis='x',labelrotation=0,pad=3,length=2,direction='in')
gca().set_xticklabels([])
plot([0,1],[0.95,0.95],color='k',linewidth=1)
plt.annotate('**',(0.38,0.87),size=xlarge)



#Panel Labels
plt.annotate('A',(0.025,0.925),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.205,0.925),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.44,0.925),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('D',(0.635,0.925),xycoords='figure fraction',size=large, fontweight='bold')


#fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Figure5.pdf',dpi=600, format='pdf',transparent=True)
