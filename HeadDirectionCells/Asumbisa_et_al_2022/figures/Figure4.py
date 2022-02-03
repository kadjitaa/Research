# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:12:42 2022

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

gsA=fig.add_gridspec(2,2,left=0.07,right=0.2,top=0.87,bottom=0.7)

###############################################################################
#A left, Polar plot- mouse head with and without whiskers
###############################################################################

rd_tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_exposure12_tc_rdWHIS.npy',allow_pickle=True)
gn_tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_exposure12_tc_gnWHIS.npy',allow_pickle=True)


for i,x in enumerate(['exp2','exp1']) :
    fig.add_subplot(gsA[0,i],projection='polar')
    plot(gn_tc[0][x][15],linewidth=1.5,c='k',zorder=5)
    gca().fill(gn_tc[0][x][15],'white',zorder=4)

    gca().set_aspect('equal')
    
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])

    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    hz=str(round(gn_tc[0][x][15].max(),1))+'Hz'
    plt.annotate(hz,(26,27),xycoords='axes points', size=xsmall,zorder=6)
    if i==0:
        plt.annotate('C'+str(i+1),(-13,21),xycoords='axes points', size=small)

       

for i,x in enumerate(['exp1','exp2']) :
    fig.add_subplot(gsA[1,i],projection='polar')
    plot(rd_tc[0][x][18],linewidth=1.5,c='k',zorder=5)
    gca().fill(rd_tc[0][x][15],'white',zorder=4)

    gca().set_aspect('equal')
    
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])

    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    hz=str(round(rd_tc[0][x][18].max(),1))+'Hz'
    plt.annotate(hz,(26,27),xycoords='axes points', size=xsmall)
    
    if i==0:
         plt.annotate('C'+str(i+2),(-13,21),xycoords='axes points', size=small)

       
#################################################################################    
#A right, stats for WhisNoWhis
#################################################################################

gsA1=fig.add_gridspec(2,2,left=0.25,right=0.38,top=0.915,bottom=0.71,wspace=1)
whisN=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_Whiskers3.npy',allow_pickle=True)

#B1 Means-merge
fig.add_subplot(gsA1[0])
expC='means'
x=pd.concat((whisN[1][expC],whisN[2][expC]),axis=0)
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_yticks([0,20,40]);gca().set_yticklabels(['0','20','40'],size=small)
#gca().set_ylim([0,1.1])
remove_box()
plot([0,1],[41,41],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('n.s',(0.35,42),size=small)
gca().set_ylabel('Mean rate (Hz)',fontsize=small,labelpad=0.5)
#gca().set_aspect('1.4')
plt.setp(gca().get_xticklabels(), visible=False)
tickParams()

#B1 Peaks-merge
fig.add_subplot(gsA1[1])
expC='peaks'
x=pd.concat((whisN[1][expC],whisN[2][expC]),axis=0)
plt.plot([0,1],[x.Whis,x.noWhis],color='gray', alpha=0.5, linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.Whis,x.noWhis], s=1, color='k',zorder=3)
plt.plot([0,1],[median(x.Whis),median(x.noWhis)],color='red',linewidth=1.5,zorder=2)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
#gca().set_ylim([0,1.1])
remove_box()
plot([0,1],[123,123],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.Whis,x.noWhis)[1])
plt.annotate('**',(0.36,114),size=xlarge)
gca().set_ylabel('Peak rate (Hz)',fontsize=small,labelpad=0.5)
#gca().set_aspect('1.4')
plt.setp(gca().get_xticklabels(), visible=False)
gca().set_yticks([0,60,120])
gca().set_yticklabels(['0','60','120'],size=small)
tickParams()


#VectorLength-merge
fig.add_subplot(gsA1[2])
expC='mvl'
x=pd.concat((whisN[1][expC],whisN[2][expC]),axis=0)
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
x=pd.concat((whisN[1][expC],whisN[2][expC]),axis=0)
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



###############################################################################
#B1) Polar plots - Floor rotation
################################################################################

gsB=fig.add_gridspec(2,2,left=0.435,right=0.565,top=0.87,bottom=0.7)
flr_180=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_floorRot180_tc_rdGn.npy',allow_pickle=True)
rd_idx=[29,9,13,24]

for i,x in enumerate(rd_idx):
    fig.add_subplot(gsB[i],projection='polar')
    plot(flr_180[0]['exp1'][x],linewidth=1.5,c='gray',label='Standard',zorder=5)
    plot(flr_180[0]['exp2'][x],linewidth=1.5,c='k',linestyle=(0,(3,3)),label='180\xb0 Floor rot.')

    gca().fill(flr_180[0]['exp1'][x],'white',zorder=4)
    
    gca().set_aspect('equal')
    
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    
legend(bbox_to_anchor=(0, 2.9), loc='upper right',ncol=1,frameon=False,columnspacing=1,handlelength=2,handletextpad=0.2)


###############################################################################
#B2) stats predicted vs observed pfd
################################################################################

gsB1=fig.add_gridspec(1,1,left=0.58,right=0.77,top=0.85,bottom=0.725)

fig.add_subplot(gsB1[0])
pfdsAB=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_flrRotBLIND_ver2.npy',allow_pickle=True)
x=pfdsAB[0].obs; x1=pfdsAB[1].obs
y=pfdsAB[0].pred; y1=pfdsAB[1].pred

plt.scatter(x,y,c='gray',alpha=0.7,s=5,edgecolor='k',label='rd1')
plt.scatter(x1,y1,c='gray',alpha=0.7,s=5,edgecolor='k',label="$\mathregular{Gnat2/1^{mut}}$")
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

x_=pd.concat((x,x1),axis=0)
y_=pd.concat((y,y1),axis=0)
p=scipy.stats.pearsonr(x_.astype('float'),y_.astype('float'))[0]
p1=scipy.stats.linregress(x_.astype('float'),y_.astype('float'))[0]
plt.annotate('r = '+ str(round(p,2))+', '+'p< 0.0001',(pi-0.6,0),size=xsmall)
plt.annotate('slope = '+str(round(p1,2)),(pi-0.6,1),size=xsmall)


###############################################################################
#C) OSN chamber + path plots     #KA56---treated     #KA80----ctrl
###############################################################################

gsC=fig.add_gridspec(1,2,left=0.05,right=0.56,top=0.62,bottom=0.45,wspace=0)
ctl_path=pd.read_csv(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\KA80_201010_gnat_ctrl.csv',delimiter=r'[" "]')
trtd_path=pd.read_csv(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\KA56_200312_rd_treated.csv',delimiter=r'[" "]')

fig.add_subplot(gsC[0])
plot(ctl_path.X,ctl_path.Y,color='darkgray',alpha=0.8)
neut,avers=makeODTarena()
gca().legend([neut,avers],['Neutral odor','Aversive odor'],loc='upper left',fontsize=xsmall,markerscale=0.5,bbox_to_anchor=(0.637, 0.96))
gca().set_aspect('equal')
title('Blind control',size=med,y=0.99)

fig.add_subplot(gsC[1])
plot(trtd_path.X,trtd_path.Y, color='darkgray',alpha=0.8)
a,b=makeODTarena()
gca().set_aspect('equal')
title('Olfaction ablated',size=med,y=0.99)



###############################################################################
#D) OSN Chamber Preference
###############################################################################

gsD=fig.add_gridspec(1,1,left=0.625,right=0.725,top=0.65,bottom=0.46)
ax1=fig.add_subplot(gsD[0])
data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_ODT3.npy',allow_pickle=True)
rd_odt=data[1]['ctrl']; rd_odt1=data[1]['treated']
gn_odt=data[2]['ctrl']; gn_odt1=data[2]['treated']

intact=np.concatenate((array(data[1]['ctrl']),array(data[2]['ctrl'])))
osn=np.concatenate((array(data[1]['treated']),array(data[2]['treated'])))

vp=ax1.violinplot([list(intact),list(osn)],showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([intact,osn],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(2):
   vp['bodies'][i].set_facecolor('lightgray')
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


gca().set_ylabel('Chamber preference \n (Neutral side - Aversive side)',labelpad=3)
gca().set_xticks([1,2])
gca().set_xticklabels(['Blind\ncontrol','Olfaction\nablated'])
plt.tight_layout()
remove_box()
tickParams()
gca().set_ylim(-1,1)
gca().set_yticks([-1,0,1])

gsC1=fig.add_gridspec(1,1,left=0.63,right=0.71,top=0.70,bottom=0.65)
axC1=fig.add_subplot(gsC1[0])
plot([0.25,0.87],[0.1,0.1],linewidth=1,color='k')
plt.annotate('**',(0.48,0.027),size=xlarge)
gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC1.axis('off')



###############################################################################
#E OSN Baseline - Post OSN ablation
###############################################################################
gsE=fig.add_gridspec(2,6,left=0.07,right=0.55,top=0.36,bottom=0.17)

rd1_osnDays=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig3_OSN_tc_rd2DayMerge.npy',allow_pickle=True)#[0,2,3,8,1,6]

pre=rd1_osnDays[0]['pre_tc']
post= rd1_osnDays[0]['post_tc']
id1=[0,2,3,8,1,6]
for i,x in enumerate(id1):
    fig.add_subplot(gsE[0,i],projection='polar')
    plot(pre[x],linewidth=1.5,c='k',zorder=5)
    gca().fill(pre[x],'white',zorder=4)

    if i==0:
        ylabel('Baseline',y=0.5,labelpad=5)
    else:
        pass
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])

    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
        
    gca().set_aspect('equal')
    hz=str(round(pre[x].max(),1))+'Hz'
    plt.annotate(hz,(33,18),xycoords='axes points', size=xsmall,zorder=6)
    plt.annotate('C'+str(i+1),(21,60),xycoords='axes points', size=small)
      
    
for i,x in enumerate(id1):
    fig.add_subplot(gsE[1,i],projection='polar')
    plot(post[x],linewidth=1.5,c='k',zorder=5)
    gca().fill(post[x],'white',zorder=4)

    if i==0:
        ylabel('Olfaction\nablated',y=0.5,labelpad=2)
    else:
        pass
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])

    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
        
    hz=str(round(post[x].max(),1))+'Hz'
    plt.annotate(hz,(18,27),xycoords='axes points', size=xsmall,zorder=6)
    



###############################################################################
#F ADN units characterised as HD cells 
###############################################################################

gsF=fig.add_gridspec(1,1,left=0.625,right=0.725,top=0.39,bottom=0.20)
ax2=fig.add_subplot(gsF[0])
count_OSN=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_hdsDistributionLightDark-OSN_4_ver2.npy',allow_pickle=True)
hd_l=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionLight3Strains_ver3.npy',allow_pickle=True)


rd_osn=count_OSN[1]['hd_%']+count_OSN[2]['hd_%']
rd_intact=hd_l[1]['hd_%']+hd_l[2]['hd_%']

vp=ax2.violinplot([list(rd_intact),list(rd_osn)],showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([rd_intact,rd_osn],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

for i in range(2):
   vp['bodies'][i].set_facecolor('lightgray')
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

gca().set_yticks([0,25,50,75,100])
ylim(-2,100)
gca().set_xticks([1,2])
gca().set_xticklabels(['Blind\ncontrol', 'Olfaction\nablated'])
gca().set_ylabel('ADn units characterised as \nHD cells (%)',labelpad=-1)
remove_box()
tickParams()


gsC2=fig.add_gridspec(1,1,left=0.63,right=0.71,top=0.44,bottom=0.39)
axC2=fig.add_subplot(gsC2[0])
plot([0.25,0.87],[0.1,0.1],linewidth=1,color='k')
plt.annotate('**',(0.48,0.027),size=xlarge)
gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC2.axis('off')


#Panel Labels
plt.annotate('A',(0.045,0.93),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.415,0.93),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.045,0.65),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('D',(0.57,0.65),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('E',(0.045,0.40),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('F',(0.57,0.40),xycoords='figure fraction',size=large, fontweight='bold')


# fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Figure4.pdf',dpi=600, format='pdf',transparent=True)


