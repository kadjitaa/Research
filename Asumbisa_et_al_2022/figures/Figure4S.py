# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:36:21 2022

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

plt.close('all')
fig=plt.figure(figsize=(11,8.5))
bkAx=bkgrid() #helper fxn to set grid in background


######################################################################################
#A
gsA1=fig.add_gridspec(2,2,left=0.11,right=0.24,top=0.915,bottom=0.71,wspace=1)
whisN=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_Whiskers3.npy',allow_pickle=True)
######################################################################################

fig.add_subplot(gsA1[0])
expC='means'
x=whisN[1][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_yticks([0,20,40]);gca().set_yticklabels(['0','20','40'],size=small)
remove_box()
plot([0,1],[41,41],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('n.s',(0.35,42),size=small)
gca().set_ylabel('Mean rate (Hz)',fontsize=small,labelpad=0.5)
plt.setp(gca().get_xticklabels(), visible=False)
tickParams()


fig.add_subplot(gsA1[1])
expC='peaks'
x=whisN[1][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
remove_box()
plot([0,1],[123,123],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('**',(0.36,114),size=xlarge)
gca().set_ylabel('Peak rate (Hz)',fontsize=small,labelpad=0.5)
plt.setp(gca().get_xticklabels(), visible=False)
gca().set_yticks([0,60,120])
gca().set_yticklabels(['0','60','120'],size=small)
tickParams()

fig.add_subplot(gsA1[2])
expC='mvl'
x=whisN[1][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticklabels(['Before','After'])
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)

gca().set_yticks([0,0.5,1]);gca().set_yticklabels(['0','.5','1'],size=small)
gca().set_ylim([0,1.1])
remove_box(2)
plot([0,1],[1,1],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('n.s',(0.34,1.03),size=small)
gca().set_ylabel('Vector length (r)',fontsize=small,labelpad=0.5)
tickParams()
plt.tick_params(axis='x',labelrotation=0,pad=3,length=2)


fig.add_subplot(gsA1[3])
expC='info'
x=whisN[1][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticklabels(['Before','After'])
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)

gca().set_yticks([0,1,2,3]);gca().set_yticklabels(['0','1','2','3'])
plot([0,1],[3,3],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('n.s',(0.34,3.08),size=small)
gca().set_ylabel('Info. (bits/spk)',labelpad=0.5)
remove_box()
tickParams()
plt.tick_params(axis='x',labelrotation=0,pad=3,length=2)


#################################################################################
#B
#################################################################################

gsB1=fig.add_gridspec(1,1,left=0.31,right=0.4,top=0.85,bottom=0.718)
pfdsAB=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_flrRotBLIND_ver2.npy',allow_pickle=True)

fig.add_subplot(gsB1[0])

x=pfdsAB[0].obs; #x1=pfdsAB[1].obs
y=pfdsAB[0].pred; #y1=pfdsAB[1].pred

plt.scatter(x,y,c='gray',alpha=0.7,s=5,edgecolor='k',label='rd1')
gca().set_aspect('equal')
gca().set_ylim([-0.5,8.5]); gca().set_yticks([0,pi,2*pi])#gca().set_yticks([0,5,10])
gca().set_yticklabels(['0','180\u00b0','360\u00b0'])

gca().set_xlim([-0.5,8.5]); gca().set_xticks([0,pi,2*pi])
gca().set_xticklabels(['0','180\u00b0','360\u00b0'])

plt.plot([0, 1], [0, 1], transform=gca().transAxes,color='red', linestyle=(0,(3,3)),linewidth=1,zorder=3)

gca().set_xlabel('Observed mean PFD')
gca().set_ylabel('Predicted mean PFD',labelpad=-0.3)
remove_box()

tickParams()
p=scipy.stats.pearsonr(x.astype('float'),y.astype('float'))[0]
p1=scipy.stats.linregress(x.astype('float'),y.astype('float'))[0]
plt.annotate('r = '+ str(round(p,2))+', '+'p< 0.0001',(pi-0.6,0),size=xsmall)
plt.annotate('slope = '+ str(round(p1)),(pi-0.6,1),size=xsmall)



#################################################################################
#C
#################################################################################

gsE=fig.add_gridspec(3,2,left=0.43,right=0.55,top=0.89,bottom=0.71)
rd1_osn_tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_OSN_tc_rd.npy',allow_pickle=True)#id1=[4,6,8,9,19,14],id2=[0,1,2,3,4,8]

pre=rd1_osn_tc[0]['intact']
post= rd1_osn_tc[1]['osn']

id1=[4,9,19]
for i,x in enumerate(id1):
    fig.add_subplot(gsE[i,0],projection='polar')
    plot(pre[x],linewidth=1.5,c='#44AA99',zorder=6)
    if i ==0:
        title('Control')
    gca().fill(pre[x],c='white',zorder=5)

    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
        
    hz=str(round(pre[x].max(),1))+'Hz'    
    plt.annotate(hz,(27,28),xycoords='axes points', size=xsmall)


id1=[0,1,2]
for i,x in enumerate(id1):
    fig.add_subplot(gsE[i,1],projection='polar')
    #subplot(6,1,i+1,projection='polar')
    plot(post[x],linewidth=1.5,c='#44AA99',zorder=6)
    if i==0:
        title('Olfaction\nablated')
    gca().fill(post[x],c='white',zorder=5)

    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    hz=str(round(pre[x].max(),1))+'Hz'
    plt.annotate(hz,(27,28),xycoords='axes points', size=xsmall)


#################################################################################
#Panle D
gsF=fig.add_gridspec(1,1,left=0.63,right=0.73,top=0.89,bottom=0.725)
gsC2=fig.add_gridspec(1,1,left=0.63,right=0.75,top=0.94,bottom=0.89)

count_OSN=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_hdsDistributionLightDark-OSN_4_ver2.npy',allow_pickle=True)
hd_l=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionLight3Strains_ver3.npy',allow_pickle=True)
#################################################################################

ax2=fig.add_subplot(gsF[0])

rd_osn=count_OSN[1]['hd_%']
rd_intact=hd_l[1]['hd_%']

vp=ax2.violinplot([list(rd_intact),list(rd_osn)],showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([rd_intact,rd_osn],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(2):
   vp['bodies'][i].set_facecolor('#44AA99')
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


gca().set_yticks([0,25,50,75,100])
ylim(-2,100)
gca().set_xticks([1,2])
gca().set_xticklabels(['Control', 'Olfaction\nablated'])
gca().set_ylabel('ADn units characterised as \nHD cells (%)',labelpad=-1)
remove_box()
tickParams()

axC2=fig.add_subplot(gsC2[0])

gca().set_ylim(0,0.8)
gca().set_xlim(0,1)
axC2.axis('off')

plot([0.214,0.63],[0.1,0.1],linewidth=1,color='k')
plt.annotate('**',(0.37,0.027),size=xlarge)



#################################################################################
#I
#################################################################################
gsA2=fig.add_gridspec(1,1,left=0.79,right=0.91,top=0.89,bottom=0.725)
gsC2=fig.add_gridspec(1,1,left=0.79,right=0.95,top=0.94,bottom=0.89)

dat1=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_cueRots3_Gain_ver3.npy',allow_pickle=True)
dat2=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_flrRots3_Gain_ver3.npy',allow_pickle=True)

a=dat1[0]
d=dat2[1]+dat2[2]

bins=np.linspace(0,1.2,20)

ax1=fig.add_subplot(gsA2[0])
vp=gca().violinplot([list(a),list(d)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([a,d],widths=0.13,positions=[0,1],showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#907398']
for i in range(2):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

gca().set_ylabel('Gain modulation',labelpad=1)
gca().set_xticks([0,1])
gca().set_xticklabels([r'WT$_L$'+'\nVisual cue rot.','Blind\nFloor rot.'])
gca().set_yticks([0,0.5,1])
gca().set_yticklabels(['0','0.5','1'])
ylim(-0.2,1.2)

remove_box()
tickParams()

axC2=fig.add_subplot(gsC2[0])
gca().set_ylim(0,0.8)
gca().set_xlim(0,1)
axC2.axis('off')
plot([0.188,0.56],[0.1,0.1],linewidth=1,color='k')
plt.annotate('n.s',(0.35,0.17),size=small)



#GNAT#
#################################################################################
#E
#################################################################################

gsA2=fig.add_gridspec(2,2,left=0.11,right=0.24,top=0.64,bottom=0.43,wspace=1)
whisN=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_Whiskers3.npy',allow_pickle=True)

fig.add_subplot(gsA2[0])
expC='means'
x=whisN[2][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_yticks([0,20,40]);gca().set_yticklabels(['0','20','40'],size=small)
remove_box()
plot([0,1],[41,41],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('n.s',(0.35,42),size=small)
gca().set_ylabel('Mean rate (Hz)',fontsize=small,labelpad=0.5)
plt.setp(gca().get_xticklabels(), visible=False)
tickParams()

fig.add_subplot(gsA2[1])
expC='peaks'
x=whisN[2][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
remove_box()
plot([0,1],[123,123],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('**',(0.36,114),size=xlarge)
gca().set_ylabel('Peak rate (Hz)',fontsize=small,labelpad=0.5)
plt.setp(gca().get_xticklabels(), visible=False)
gca().set_yticks([0,60,120])
gca().set_yticklabels(['0','60','120'],size=small)
tickParams()

fig.add_subplot(gsA2[2])
expC='mvl'
x=whisN[2][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticklabels(['Before','After'])
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_yticks([0,0.5,1]);gca().set_yticklabels(['0','.5','1'],size=small)
gca().set_ylim([0,1.1])
remove_box(2)
plot([0,1],[1,1],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('n.s',(0.34,1.03),size=small)
gca().set_ylabel('Vector length (r)',fontsize=small,labelpad=0.5)
tickParams()
plt.tick_params(axis='x',labelrotation=0,pad=3,length=2)


fig.add_subplot(gsA2[3])
expC='info'
x=whisN[2][expC]
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticklabels(['Before','After'])
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)

gca().set_yticks([0,1,2,3]);gca().set_yticklabels(['0','1','2','3'])
plot([0,1],[3,3],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('n.s',(0.34,3.08),size=small)
gca().set_ylabel('Info. (bits/spk)',labelpad=0.5)
remove_box()
tickParams()
plt.tick_params(axis='x',labelrotation=0,pad=3,length=2)


#################################################################################
#F
#################################################################################
gsB1=fig.add_gridspec(1,1,left=0.31,right=0.4,top=0.565,bottom=0.4418)
fig.add_subplot(gsB1[0])
pfdsAB=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_flrRotBLIND_ver2.npy',allow_pickle=True)
x=pfdsAB[1].obs; 
y=pfdsAB[1].pred;

plt.scatter(x,y,c='gray',alpha=0.7,s=5,edgecolor='k',label='rd1')
gca().set_aspect('equal')
gca().set_ylim([-0.5,8.5]); gca().set_yticks([0,pi,2*pi])#gca().set_yticks([0,5,10])
gca().set_yticklabels(['0','180\u00b0','360\u00b0'])

gca().set_xlim([-0.5,8.5]); gca().set_xticks([0,pi,2*pi])
gca().set_xticklabels(['0','180\u00b0','360\u00b0'])

plt.plot([0, 1], [0, 1], transform=gca().transAxes,color='red', linestyle=(0,(3,3)),linewidth=1,zorder=3)

gca().set_xlabel('Observed mean PFD')
gca().set_ylabel('Predicted mean PFD',labelpad=-0.3)
remove_box()
tickParams()
p=scipy.stats.pearsonr(x.astype('float'),y.astype('float'))[0]
p1=scipy.stats.linregress(x.astype('float'),y.astype('float'))[0]
plt.annotate('r = '+ str(round(p,2))+', '+'p< 0.0001',(pi-0.6,0),size=xsmall)
plt.annotate('slope = '+ str(round(p1,2)),(pi-0.6,1),size=xsmall)


#################################################################################
#G
#################################################################################

gsE=fig.add_gridspec(3,2,left=0.43,right=0.55,top=0.61,bottom=0.43)
gn_osn_tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_OSN_tc_gn.npy',allow_pickle=True)#id1=[3,5,7] id2=[2,3,0]

pre=gn_osn_tc[0]['intact']
post= gn_osn_tc[1]['osn']

id1=[3,5,7]
for i,x in enumerate(id1):
    fig.add_subplot(gsE[i,0],projection='polar')
    plot(pre[x],linewidth=1.5,c='#DC3C95',zorder=6)
    if i ==0:
        title('Control')
    gca().fill(pre[x],c='white',zorder=5)

    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    hz=str(round(pre[x].max(),1))+'Hz'    
    plt.annotate(hz,(27,28),xycoords='axes points', size=xsmall)

id1=[2,3,0]
for i,x in enumerate(id1):
    fig.add_subplot(gsE[i,1],projection='polar')
    plot(post[x],linewidth=1.5,c='#DC3C95',zorder=6)
    if i==0:
        title('Olfaction\nablated')
    gca().fill(post[x],c='white',zorder=5)

    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    hz=str(round(pre[x].max(),1))+'Hz'
    plt.annotate(hz,(27,28),xycoords='axes points', size=xsmall)


###############################################################################
#H
###############################################################################

gsF=fig.add_gridspec(1,1,left=0.63,right=0.73,top=0.605,bottom=0.445)
ax2=fig.add_subplot(gsF[0])

count_OSN=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_hdsDistributionLightDark-OSN_4_ver2.npy',allow_pickle=True)
hd_l=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionLight3Strains_ver3.npy',allow_pickle=True)

rd_osn=count_OSN[2]['hd_%']
rd_intact=hd_l[2]['hd_%']

vp=ax2.violinplot([list(rd_intact),list(rd_osn)],showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([rd_intact,rd_osn],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(2):
   vp['bodies'][i].set_facecolor('#DC3C95')
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


gca().set_yticks([0,25,50,75,100])
ylim(-2,100)
gca().set_xticks([1,2])
gca().set_xticklabels(['Control', 'Olfaction\nablated'])
gca().set_ylabel('ADn units characterised as \nHD cells (%)',labelpad=-1)
remove_box()
tickParams()

gsC2=fig.add_gridspec(1,1,left=0.63,right=0.75,top=0.655,bottom=0.605)
axC2=fig.add_subplot(gsC2[0])
gca().set_ylim(0,0.8)
gca().set_xlim(0,1)
axC2.axis('off')
plot([0.214,0.63],[0.1,0.1],linewidth=1,color='k')
plt.annotate('**',(0.36,0.022),size=xlarge)


###############################################################################
#J
###############################################################################
gsF1=fig.add_gridspec(1,1,left=0.79,right=0.91,top=0.605,bottom=0.445)
ax2=fig.add_subplot(gsF1[0])

dats=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig5S_OSN_whiskersVSnWhiskers.npy',allow_pickle=True)
a=dats[0]
b=dats[1]

vp=ax2.violinplot([list(a),list(b)],showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([a,b],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(2):
   vp['bodies'][i].set_facecolor('lightgray')
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

gca().set_ylim(0,1)
gca().set_ylabel('Vector length (r)',labelpad=1)
gca().set_yticks([0,0.5,1])
gca().set_yticklabels([0,'0.5',1])
gca().set_xticks([1,2])
gca().set_xticklabels(['Intact\nwhiskers','Ablated\nwhiskers'])

plot([1,2],[0.9,0.9],linewidth=1,color='k')
plt.annotate('n.s',(1.4,0.93),size=small)
title ('Olfaction ablated')
remove_box()
tickParams()


#Panel Labels
plt.annotate('A',(0.05,0.93),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.265,0.93),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.42,0.93),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('D',(0.58,0.93),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('I',(0.75,0.93),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('E',(0.05,0.65),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('F',(0.265,0.65),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('G',(0.42,0.65),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('H',(0.58,0.65),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('J',(0.75,0.65),xycoords='figure fraction',size=large, fontweight='bold')


plt.annotate('Whisker ablation',(0.13,0.93),xycoords='figure fraction',size=med,fontweight='normal')
plt.annotate('Whisker ablation',(0.13,0.655),xycoords='figure fraction',size=med,fontweight='normal')
plt.annotate('rd1',(0.06,0.80),xycoords='figure fraction',size=large,rotation=90, color='#44AA99',fontweight='bold')
plt.annotate(r'Gnat1/2$^{mut}$',(0.06,0.48),xycoords='figure fraction',size=large,rotation=90, color='#DC3C95',fontweight='bold')

bkAx.axis('off')

#fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Figure4S.pdf',dpi=600, format='pdf',transparent=True)
