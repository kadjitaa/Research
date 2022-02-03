# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:03:29 2022

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
bkAx=bkgrid() #helper fxn to set grid in background


####################################################################################
## PANEL A
gsA=fig.add_gridspec(1,5,left=0.1,right=0.73,top=0.89,bottom=0.76,wspace=0.4)

####################################################################################

data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_OSN_WTd_Blind_ver2.npy',allow_pickle=True)
data2=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_LightDark3.npy',allow_pickle=True)  
ndat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_stnd_WTd_LD_session.npy',allow_pickle=True).item()

##############################################################################
###Mean Frate ##########################################################
con='means'

a2=ndat[con]
b= np.concatenate((data[1][con],data[2][con]))

ax1=fig.add_subplot(gsA[0])
vp=ax1.violinplot([list(a2.values.flatten()),list(b.flatten())],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([a2.values.flatten(),b.flatten()],positions=[0,1],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','#986D97']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


remove_box()
gca().set_ylabel('Mean rate (Hz)',labelpad=0)
plot([0,1],[33,33],color='k',linewidth=1)
ylim(0,35)
scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))

plt.annotate('n.s',(0.4,33.5),size=small)
tickParams()
gca().set_xticklabels([])

wt_patch = mpatches.Patch(color='darkgray', label=r'WT$_D$')
blind_patch = mpatches.Patch(color='#986D97', label='Blind: Olfaction ablated')

plt.legend(handles=[wt_patch,blind_patch],bbox_to_anchor=(-0.01,1.29),loc='upper left',ncol=2,frameon=True,labelspacing=0.4,handlelength=2,handletextpad=0.4)



##############################################################################
###Peak Frate ##########################################################
con='peaks'

a2=ndat[con]
b= np.concatenate((data[1][con],data[2][con]))

ax2=fig.add_subplot(gsA[1])

vp=ax2.violinplot([list(a2.values.flatten()),list(b.flatten())],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([a2.values.flatten(),b.flatten()],positions=[0,1],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','#986D97']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
gca().set_ylabel('Peak rate (Hz)',labelpad=0)
plot([0,1],[68,68],color='k',linewidth=1)
ylim(0,70)
scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))
plt.annotate('*',(0.45,65),size=xlarge)
gca().set_xticklabels([])
tickParams()


##############################################################################
### Stability ##########################################################
con='corr'

a2=ndat[con]
b= np.concatenate((data[1][con],data[2][con]))

ax3=fig.add_subplot(gsA[2])
vp=ax3.violinplot([list(a2.values.flatten()),list(b.flatten())],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([a2.values.flatten(),b.flatten()],positions=[0,1],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','#986D97']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
gca().set_ylabel('Stability (r)',labelpad=0)
plot([0,1],[0.99,0.99],color='k',linewidth=1)
scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))
plt.annotate('**',(0.35,0.9),size=xlarge)
ylim(-1,1)
gca().set_yticks([-1,-0.5,0,0.5,1]);
gca().set_yticklabels(['-1','-0.5','0','0.5','1'])
gca().set_xticklabels([])
tickParams()



##############################################################################
### Width ##########################################################
con='width'

a2=ndat[con]
b= np.concatenate((data[1][con],data[2][con]))

ax4=fig.add_subplot(gsA[3])
vp=ax4.violinplot([list(a2.values.flatten()),list(b.flatten())],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([a2.values.flatten(),b.flatten()],positions=[0,1],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','#986D97']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
gca().set_ylabel('Tuning width',labelpad=0)
plot([0,1],[5.5,5.5],color='k',linewidth=1)
scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))
plt.annotate('**',(0.35,5.3),size=xlarge)
gca().set_yticks([deg2rad(0),deg2rad(90),deg2rad(180),deg2rad(270),deg2rad(360)]);
gca().set_yticklabels(['0\u00b0','90\u00b0','180\u00b0','270\u00b0','360\u00b0'])
gca().set_xticklabels([])
tickParams()


##############################################################################
###Info ##########################################################
con='info'

a2=ndat[con]
b= np.concatenate((data[1][con],data[2][con]))

ax5=fig.add_subplot(gsA[4])
vp=ax5.violinplot([list(abs(a2.values.flatten())),list(abs(b.flatten()))],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([abs(a2.values.flatten()),abs(b.flatten())],positions=[0,1],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','#986D97']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
gca().set_ylabel('Info (bits/spk)',labelpad=0)
plot([0,1],[1.3,1.3],color='k',linewidth=1)
ylim(0,1.5)
gca().set_yticks([0,0.3,0.6,0.9,1.2,1.5])
gca().set_yticklabels(['0','0.3','0.6','0.9',1.2,1.5])
scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))
plt.annotate('**',(0.35,1.25),size=xlarge)
gca().set_xticklabels([])
tickParams()



####################################################################################
## PANEL B1
gsB=fig.add_gridspec(3,1,left=0.1,right=0.26,top=0.67,bottom=0.33,hspace=0.4)
####################################################################################
dat=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_flrRots3_Gain_ver3.npy',allow_pickle=True)

a=dat[0]
b=dat[1]
c=dat[2]
bins=np.linspace(0,1.2,20)

colr=['darkgray','#44AA99','#DC3C95'];mice=[r'WT$_D$','rd1',r'Gnat1/2$^{mut}$']
for i,x in enumerate([a,b,c]):
    fig.add_subplot(gsB[i])
    hist(x,bins, histtype='stepfilled',density=True, edgecolor='k',color=colr[i],linewidth=1,label=mice[i],alpha=0.9)
    remove_box()
    gca().set_xticks([0,0.5,1])
    gca().set_xticklabels(['0',0.5,1])
    legend(loc='upper right',frameon=False,handlelength=1.1,handletextpad=0.3,ncol=1)
    if i==1:
        gca().set_ylabel('Normalized counts',labelpad=-0.2)
    plt.annotate('Median= '+str(round(median(x),2)),(0.2,3.19),size=xsmall)
    if i ==2:
        gca().set_xlabel('Gain (Floor control)',labelpad=0,x=0.42)

    tickParams()
    
####################################################################################
## PANEL B2
flr_180wt=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_floorRot_tc_wtD.npy',allow_pickle=True)
flr_180bld=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_floorRot180_tc_rdGn.npy',allow_pickle=True)    
gsC=fig.add_gridspec(3,3,left=0.3,right=0.48,top=0.63,bottom=0.35,hspace=1.5,wspace=0.6)
####################################################################################

wt_idx=[7,8,10]

for i,x in enumerate(wt_idx):
    fig.add_subplot(gsC[0,i])#,projection='polar')
    plot(flr_180wt[0]['floorA'][x],linewidth=1.5,c='k')
    lb=plot(flr_180wt[0]['floorB'][x],linewidth=1.5,c='grey',linestyle=(0,(3,3)),label='180\xb0 Floor rot.')
    gca().set_xticks([0,2*pi])
    gca().set_xticklabels(['0','360\u00b0'])
    if i==0:
        ylim(0,22)
        gca().set_ylabel('Firing rate',labelpad=0)
    if i==1:
        xlabel('Head direction',labelpad=0)
    remove_box()
    tickParams()
legend(bbox_to_anchor=(-0.32, 1.57), loc='upper right',frameon=True) 
    
rd_idx=[20,10,12]
for i,x in enumerate(rd_idx):
    fig.add_subplot(gsC[1,i])#,projection='polar')
    #title('#'+str(i))
    plot(flr_180bld[0]['exp1'][x],linewidth=1.5,c='k')
    lb1=plot(flr_180bld[0]['exp2'][x],linewidth=1.5,c='#44AA99',linestyle=(0,(3,3)),label='180\xb0 Floor rot.')
    gca().set_xticks([0,2*pi])
    gca().set_xticklabels(['0','360\u00b0'])
    if i==1:
        xlabel('Head direction',labelpad=0)
    if i==0:
        gca().set_ylabel('Firing rate',labelpad=0)
    remove_box()
    tickParams()
legend(bbox_to_anchor=(-0.32, 1.57), loc='upper right',frameon=True) 

gn_idx=[4,7,20]
for i,x in enumerate(gn_idx):
    fig.add_subplot(gsC[2,i])
    plot(flr_180bld[1]['exp1'][x],linewidth=1.5,c='k')
    lb2=plot(flr_180bld[1]['exp2'][x],linewidth=1.5,c='#DC3C95',linestyle=(0,(3,3)),label='180\xb0 Floor rot.')
    gca().set_xticks([0,2*pi])
    gca().set_xticklabels(['0','360\u00b0'])
    if i==0:
        gca().set_ylabel('Firing rate',labelpad=0)
    if i==1:
        xlabel('Head direction',labelpad=0)
    remove_box()
    tickParams()
legend(bbox_to_anchor=(-0.32, 1.57), loc='upper right',frameon=True) 


#####################################################################################
### PANEL C
gsD=fig.add_gridspec(3,2,left=0.55,right=0.70,top=0.66,bottom=0.33,hspace=0.43,wspace=1)
####################################################################################
data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_OSN_wtLD_xtics.npy',allow_pickle=True).item()

con='means'
a=data[con].light.values.flatten()
b= data[con].dark.values.flatten() 
fig.add_subplot(gsD[0])

plt.plot([0,1],[a,b],color='gray', alpha=0.5, linewidth=0.8)
plt.plot([0,1],[median(a),median(b)],color='red',linewidth=1.5,zorder=2)
plt.scatter(np.zeros(len(a)),a,color='#56B4E9',s=5,zorder=3,alpha=0.4)
plt.scatter(np.ones(len(b)),b,color='k',s=5,zorder=3,alpha=0.5)
ylim(0,35)
yticks([0,10,20,30])
gca().set_ylabel('Mean rate (Hz)',labelpad=0)
gca().set_xlim(-0.1,1.1)
gca().set_xticklabels([])
remove_box()
tickParams()

gsC2=fig.add_gridspec(1,1,left=0.55,right=0.62,top=0.72,bottom=0.65)
axC2=fig.add_subplot(gsC2[0])
plot([0.057,0.644],[0.1,0.1],linewidth=1,color='k')
gca().set_ylim(0,0.5)
gca().set_xlim(0,1)
axC2.axis('off')
plt.annotate('n.s',(0.25,0.11),size=small)


con='peaks'
a=data[con].light.values.flatten()
b= data[con].dark.values.flatten()

fig.add_subplot(gsD[1])

plt.plot([0,1],[a,b],color='gray', alpha=0.5, linewidth=0.8)
plt.plot([0,1],[median(a),median(b)],color='red',linewidth=1.5,zorder=2)
plt.scatter(np.zeros(len(a)),a,color='#56B4E9',s=5,zorder=3,alpha=0.4)
plt.scatter(np.ones(len(b)),b,color='k',s=5,zorder=3,alpha=0.5)
gca().set_ylabel('Peak rate (Hz)',labelpad=0)
gca().set_xlim(-0.1,1.1)
gca().set_xticklabels([])
remove_box()
tickParams()


gsC2=fig.add_gridspec(1,1,left=0.64,right=0.71,top=0.72,bottom=0.65)
axC2=fig.add_subplot(gsC2[0])
plot([0.21,0.78],[0.1,0.1],linewidth=1,color='k')
gca().set_ylim(0,0.5)
gca().set_xlim(0,1)
axC2.axis('off')
plt.annotate('**',(0.4,0.055),size=xlarge)


con='corr'
a=data[con].light.values.flatten()
b= data[con].dark.values.flatten()
 
fig.add_subplot(gsD[2])

plt.plot([0,1],[a,b],color='gray', alpha=0.5, linewidth=0.8)
plt.plot([0,1],[median(a),median(b)],color='red',linewidth=1.5,zorder=2)
plt.scatter(np.zeros(len(a)),a,color='#56B4E9',s=5,zorder=3,alpha=0.4)
plt.scatter(np.ones(len(b)),b,color='k',s=5,zorder=3,alpha=0.5)
gca().set_ylabel('Stability (r)',labelpad=0)
gca().set_xticklabels([])
gca().set_xlim(-0.1,1.1)
gca().set_ylim(-1,1.1)
remove_box()
tickParams()


gsC2=fig.add_gridspec(1,1,left=0.55,right=0.62,top=0.59,bottom=0.52)
axC2=fig.add_subplot(gsC2[0])
plot([0.057,0.644],[0.1,0.1],linewidth=1,color='k')
gca().set_ylim(0,0.5)
gca().set_xlim(0,1)
axC2.axis('off')
plt.annotate('**',(0.25,0.055),size=xlarge)


con='width'
a=data[con].light.values.flatten()
b= data[con].dark.values.flatten()
 
fig.add_subplot(gsD[3])

plt.plot([0,1],[a,b],color='gray', alpha=0.5, linewidth=0.8)
plt.plot([0,1],[median(a),median(b)],color='red',linewidth=1.5,zorder=2)
plt.scatter(np.zeros(len(a)),a,color='#56B4E9',s=5,zorder=3,alpha=0.4)
plt.scatter(np.ones(len(b)),b,color='k',s=5,zorder=3,alpha=0.5)
gca().set_ylabel('Tuning width',labelpad=0)
gca().set_ylim([0,deg2rad(275)])
gca().set_yticks([deg2rad(0),deg2rad(90),deg2rad(180),deg2rad(270)]);
gca().set_yticklabels(['0\u00b0','90\u00b0','180\u00b0','270\u00b0'])
gca().set_xlim(-0.1,1.1)
gca().set_xticks([0,1])
gca().set_xticklabels([r'WT$_L$',r'WT$_D$'])
title('Olfaction ablated',y=-0.5,size=small)
remove_box()
tickParams()

gsC2=fig.add_gridspec(1,1,left=0.64,right=0.71,top=0.59,bottom=0.52)
axC2=fig.add_subplot(gsC2[0])
plot([0.21,0.78],[0.1,0.1],linewidth=1,color='k')
gca().set_ylim(0,0.5)
gca().set_xlim(0,1)
axC2.axis('off')
plt.annotate('**',(0.4,0.06),size=xlarge)


con='info'
a=data[con].light.values.flatten()
b= data[con].dark.values.flatten()

fig.add_subplot(gsD[4])

plt.plot([0,1],[a,b],color='gray', alpha=0.5, linewidth=0.8)
plt.plot([0,1],[median(a),median(b)],color='red',linewidth=1.5,zorder=2)
plt.scatter(np.zeros(len(a)),a,color='#56B4E9',s=5,zorder=3,alpha=0.4)
plt.scatter(np.ones(len(b)),b,color='k',s=5,zorder=3,alpha=0.5)
gca().set_ylabel('Info (bits/spk)',labelpad=0)
gca().set_ylim([0,2.2])
gca().set_yticks([0,1,2])
gca().set_xlim(-0.1,1.1)
gca().set_xticks([0,1])
gca().set_xticklabels([r'WT$_L$',r'WT$_D$'])
title('Olfaction ablated',y=-0.5,size=small)
remove_box()
tickParams()


gsC2=fig.add_gridspec(1,1,left=0.55,right=0.62,top=0.46,bottom=0.39)
axC2=fig.add_subplot(gsC2[0])

plot([0.057,0.644],[0.1,0.1],linewidth=1,color='k')
gca().set_ylim(0,0.5)
gca().set_xlim(0,1)
axC2.axis('off')
plt.annotate('**',(0.25,0.055),size=xlarge)


###############################################################################
#Panel D Distribution of HD counts
###############################################################################

gsF=fig.add_gridspec(1,1,left=0.78,right=0.89 ,top=0.65,bottom=0.45)
ax2=fig.add_subplot(gsF[0])

countsOSN=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_hdsDistributionLightDark-OSN_4_ver2.npy',allow_pickle=True)
light_counts=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_hdsDistributionLight3_ver1.npy',allow_pickle=True)
dark_counts=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_hdsDistributionDark3_ver1.npy',allow_pickle=True)

a=light_counts[0]['hd_%']   #WT_light
a1=countsOSN[0]['hd_%']     #WT_light OSN
b=dark_counts[0]['hd_%']    #WT_dark
b1=countsOSN[3]['hd_%']     #WT_dark_OSN

vp=ax2.violinplot([list(a),list(a1)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([a,a1],positions=[0,1],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['#56B4E9','#56B4E9']
alpha=[0.9,0.3]
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(alpha[i])
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
gca().set_xticklabels([])#['WTD','WTD\nOlfaction ablated'],size=small)
gca().set_ylim(-2,100)
gca().set_yticks([0,25,50,75,100])
gca().set_ylabel('ADn units characterised as\nHD cells (%)',labelpad=-2)

wtl = mpatches.Patch(color='#56B4E9', label=r'WT$_L$',alpha=0.9)
wtlosn = mpatches.Patch(color='#56B4E9', label=r'WT$_L$'+': Olfaction ablated',alpha=0.4)

plt.legend(handles=[wtl,wtlosn],bbox_to_anchor=(-0.013,0.25),loc='upper left',ncol=1,frameon=True,labelspacing=0.4,handlelength=2,handletextpad=0.4)
remove_box()
tickParams()
gca().tick_params(axis='x',pad=2, length=0)


###############################################################################
#Panel E
##############################################################################

data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_OSN_WTd_Blind_ver2.npy',allow_pickle=True)
data2=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_LightDark3.npy',allow_pickle=True)  
ndat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig4_stnd_WTd_LD_session.npy',allow_pickle=True).item()


##############################################################################
###Mean Frate ##########################################################
con='means'
a2=ndat[con].values.flatten() #WTD---autocorrs
b= np.concatenate((data[1][con].osn,data[2][con].osn)) #RD & GNAT ---osn blind autocorr
c=data[0][con].osn.values.flatten() #WTD osn

gsE=fig.add_gridspec(1,5,left=0.1,right=0.89,top=0.23,bottom=0.1,wspace=0.4)
ax1=fig.add_subplot(gsE[0])

vp=ax1.violinplot([list(a2),list(c),list(b)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2])#widths=[0.14,0,0.82]
bp=boxplot([a2,c,b],positions=[0,1,2],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','lightgray','#986D97']
for i in range(3):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
gca().set_ylabel('Mean rate (Hz)')
tickParams()
gca().set_xticklabels([])
gca().set_yticks([0,10,20,30,40])
ylim(0,40)
tickParams()
gca().set_xticklabels([])

wt_patch = mpatches.Patch(color='darkgray', label=r'WT$_D$')
wt_patch1 = mpatches.Patch(color='lightgray', label=r'WT$_D$'+': Olfaction ablated')
blind_patch = mpatches.Patch(color='#986D97', label='Blind: Olfaction ablated')
plt.legend(handles=[wt_patch,wt_patch1,blind_patch],bbox_to_anchor=(-0.01,1.29),loc='upper left',ncol=3,frameon=True,labelspacing=0.4,handlelength=2,handletextpad=0.4)


##############################################################################
###Peak Frate ##########################################################
con='peaks'

ax2=fig.add_subplot(gsE[1])

a2=ndat[con].values.flatten() #WTD---autocorrs
b= np.concatenate((data[1][con].osn,data[2][con].osn)) #RD & GNAT ---osn blind autocorr
c=data[0][con].osn.values.flatten() #WTD osn

print(scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))[1])

vp=ax2.violinplot([list(a2),list(c),list(b)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2])#widths=[0.14,0,0.82]
bp=boxplot([a2,c,b],positions=[0,1,2],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['darkgray','lightgray','#986D97']
for i in range(3):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

gca().set_ylabel('Peak rate (Hz)',labelpad=-1)
ylim(0,75)
remove_box()
scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))

gca().set_xticklabels([])
tickParams()


##############################################################################
### Stability ##########################################################
con='corr'

ax3=fig.add_subplot(gsE[2])

a2=ndat[con].values.flatten() #WTD---autocorrs
b= np.concatenate((data[1][con].osn,data[2][con].osn)) #RD & GNAT ---osn blind autocorr
c=data[0][con].osn.values.flatten() #WTD osn

print(scipy.stats.mannwhitneyu(c.astype('float'), b.astype('float'))[1])

vp=ax3.violinplot([list(a2),list(c),list(b)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2])#widths=[0.14,0,0.82]
bp=boxplot([a2,c,b],positions=[0,1,2],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(3):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

remove_box()
gca().set_ylabel('Stability (r)',labelpad=0)
ylim(-1,1)
gca().set_yticks([-1,-0.5,0,0.5,1]);
gca().set_yticklabels(['-1','-0.5','0','0.5','1'])
gca().set_xticklabels([])
tickParams()



##############################################################################
### Width ##########################################################
con='width'

ax4=fig.add_subplot(gsE[3])

a2=ndat[con].values.flatten() #WTD---autocorrs
b= np.concatenate((data[1][con].osn,data[2][con].osn)) #RD & GNAT ---osn blind autocorr
c=data[0][con].osn.values.flatten() #WTD osn

print(scipy.stats.mannwhitneyu(a2.astype('float'), b.astype('float'))[1])

vp=ax4.violinplot([list(a2),list(c),list(b)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2])#widths=[0.14,0,0.82]
bp=boxplot([a2,c,b],positions=[0,1,2],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(3):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


remove_box()
gca().set_ylabel('Tuning width',labelpad=0)
# plot([0,1],[3,3],color='k',linewidth=1)


# plt.annotate('**',(0.35,3),size=xlarge)
gca().set_yticks([deg2rad(0),deg2rad(90),deg2rad(180),deg2rad(270),deg2rad(360)]);
gca().set_yticklabels(['0\u00b0','90\u00b0','180\u00b0','270\u00b0','360\u00b0'])
gca().set_xticklabels([])
ylim([deg2rad(0),deg2rad(360)])
tickParams()



##############################################################################
###Info ##########################################################
con='info'

ax5=fig.add_subplot(gsE[4])
a2=ndat[con].values.flatten() #WTD---autocorrs
b= abs(np.concatenate((data[1][con].osn,data[2][con].osn))) #RD & GNAT ---osn blind autocorr
c=abs(data[0][con].osn.values.flatten()) #WTD osn


print(scipy.stats.mannwhitneyu(c.astype('float'), b.astype('float'))[1])

vp=ax5.violinplot([list(a2),list(c),list(b)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2])#widths=[0.14,0,0.82]
bp=boxplot([a2,c,b],positions=[0,1,2],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(3):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
gca().set_ylabel('Info (bits/spk)',labelpad=0)
gca().set_yticks([0,0.5,1,1.5])
gca().set_yticklabels(['0','0.5',1,1.5])
ylim(0,1.5)
gca().set_xticklabels([])
tickParams()


#Panel Labels
plt.annotate('A',(0.06,0.915),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.06,0.67),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.51,0.67),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('D',(0.73,0.67),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('E',(0.06,0.25),xycoords='figure fraction',size=large, fontweight='bold')

bkAx.axis('off')

#fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Figure5S.pdf',dpi=600, format='pdf',transparent=True)

