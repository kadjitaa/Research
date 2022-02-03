# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:24:16 2022

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
plt.rc('lines', linewidth=1.5)



def tickParams(axis='both'):
    gca().tick_params(axis=axis,which='major',pad=1.5, length=2)
###############################################################################


fig=plt.figure(figsize=(11,8.5))
#bkAx=bkgrid() #helper fxn to set grid in background


#############################################
# A. ARENA + HISTOLOGY
#############################################

gs1=fig.add_gridspec(1,1,left=0.04,right=0.42,bottom=0.53, top=0.75)
axA=fig.add_subplot(gs1[0])
im1= image.imread(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/Fig1A_hist.tif')
axA.imshow(im1,interpolation = 'bilinear',zorder=3)
axA.axis('off')


###########################################
# B. SPK PATH + HEATMAPS + TCURVES 
###########################################

gs3=fig.add_gridspec(3,4,left=0.47,right=0.71,bottom=0.53, top=0.75)

#Spk Path
tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_example_tcurves_rdL.npy',allow_pickle=True).item()        
ep=tc['ep']
all_spks=tc['spk']
pos=tc['pos']
#ids=tc['id']
ids=[9,12,0,4]
for i,k in enumerate(ids): 
    axB=fig.add_subplot(gs3[0,i])
    plot(pos['x'].restrict(ep),pos['z'].restrict(ep), linewidth=1, color='k', alpha=1)
    scatter(pos['x'].realign(all_spks[k].restrict(ep)), pos['z'].realign(all_spks[k].restrict(ep)),s=0.00001,color='red',alpha=0.5,zorder=3)
    sns.despine(ax=gca(),left=True, top=True, bottom=True, right=True)
    gca().set_yticks([])
    gca().set_xticks([])
    gca().set_aspect('equal')
    if i==0:
        ylabel('Spikes',size=small)
        #plt.annotate('Spikes',(pi,3),xycoords='polar', size=small,rotation=90)

    else:
        pass

#Rate map
GF, ext = computePlaceFields(all_spks, pos[['x', 'z']], ep, 100)#use 70 bins if u want to exclude outliers
for i,k in enumerate(ids):
   axB1=fig.add_subplot(gs3[1,i])
   tmp = gaussian_filter(GF[k].values,sigma = 2)
   im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'spline36')
   gca().invert_yaxis()
   gca().set_aspect('equal')
   sns.despine(ax=gca(),left=True, top=True, bottom=True, right=True)
   gca().set_yticks([])
   gca().set_xticks([])
   if i==0:
       ylabel('Rate map',size=small)

   else:
       pass


#Polar plots
for i,x in enumerate(ids):
    axB2=fig.add_subplot(gs3[2,i], projection='polar')    
    plot(tc['rd_light'][x], label=str(x),linewidth=1.5,c='k',zorder=5)
    remove_polarAx(gca(),True)
    gca().set_yticks([])
    gca().set_aspect('equal')
    hz=str(round(tc['rd_light'][x].max(),1))+'Hz'
    plt.annotate(hz,(34,33),xycoords='axes points', size=xsmall)
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    gca().fill(tc['rd_light'][x],'white',zorder=4)

    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
        
    if i==0:
        ylabel('Tuning curve')
    else:
        pass
    
    

##########################################################
# C. DISTRIBUTION OF OCCUPANCY AND PFD
##########################################################

gs5=fig.add_gridspec(2,1, left=0.72,right=0.85,bottom=0.56, top=0.75,hspace=0.8)

gsD1=fig.add_subplot(gs5[0],projection='polar')
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_position.npy',allow_pickle=True).item()    
#rd1
ry=[]
for i in range(len(dat['rd_posL'])):
    ry.extend(dat['rd_posL'][i]['ry'].values)
    
a,b,p=circular_hist( gsD1,np.array(ry),offset=0,bins=15,density=True,gaps=False)
remove_polarAx(gca(),False)
gca().set_xticklabels(['0\u00b0'," ",'90\u00b0',' ','180\u00b0'," ",'270\u00b0'])
gca().tick_params(axis='both', pad=-3)
title('Heading Dir.',y=-0.55,size=small)


##RD1 PFDs
gsD2=fig.add_subplot(gs5[1],projection='polar')
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\baseline_DAT3.npy',allow_pickle=True)     
angles0 = dat[1]['cmean']
a,b,p=circular_hist( gsD2,angles0,offset=0,bins=15,density=True,gaps=False)
remove_polarAx(gca(),False)
gca().set_xticklabels(['0\u00b0'," ",'90\u00b0',' ','180\u00b0'," ",'270\u00b0'])
gca().tick_params(axis='both', pad=-3)
title('Preferred Firing Dir.',y=-0.55,size=small)




##########################################################
# D. 1st vs 2nd Half mean PFD
##########################################################

gs6=fig.add_gridspec(2,2, left=0.05,right=0.17,bottom=0.35,top=0.47,hspace=0.6,wspace=0.6)
#1D TCURVES
tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_example_tcurves_rdL.npy',allow_pickle=True).item()        
ep=tc['ep']
all_spks=tc['spk']
pos=tc['pos']
ids=tc['id']
id_=[12,4,5,9]

eps=slidingWinEp(ep,3e+8)
tc1=computeAngularTuningCurves(all_spks,pos['ry'],nts.IntervalSet(eps.loc[0].start,eps.loc[0].end),60)[ids]
tc2=computeAngularTuningCurves(all_spks,pos['ry'],nts.IntervalSet(eps.loc[1].start,eps.loc[1].end),60)[ids]

ct=0
for i in id_:
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
        legend(bbox_to_anchor=(0.6, 1.7), loc='upper right',ncol=2,frameon=False,columnspacing=1,handletextpad=0.3,labelspacing=0.12)
    if ct==2:
        plt.ylabel('Firing rate (Hz)',y=1.2,labelpad=0)


    if i in [5,9]:
        gca().set_xlabel('')
        gca().set_xticks([0, 2*pi])
        gca().set_xticklabels([0, 2*pi])
        gca().set_xticklabels(['0\u00b0','360\u00b0'])
    ct+=1
gca().set_xlabel('Head direction',x=-0.4,labelpad=0)




gs7=fig.add_gridspec(1,1, left=0.22,right=0.34,bottom=0.35,top=0.47)
gs_dd1=fig.add_subplot(gs7[0])

#STABILITY CORR
data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_stabilityL2Eps.npy',allow_pickle=True)

x=data[1][0].T
for i in range(1,len(data[1])):
    x=pd.concat([x,data[1][i].T])

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
plt.annotate('r = '+ str(round(p,2))+', p< 0.0001',(pi-0.3,0.5),size=xsmall)
remove_box()
tickParams()



##########################################################
# E. Exposure 1 vs Exposure 2 mean PFD
##########################################################

gs8=fig.add_gridspec(2,2, left=0.40,right=0.52,bottom=0.35,top=0.47,hspace=0.6,wspace=0.6)

#example tcurves exposur1vs2
tc_12=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_exposure12_tc_rdWHIS.npy',allow_pickle=True)
id_=[20,12,10,24]
tc1=tc_12[0]['exp1']
tc2=tc_12[0]['exp2']

ct=0
for i in id_:
    fig.add_subplot(gs8[ct])

    plot(tc1[i],color='gray', label='Exposure 1',linewidth=1.5)
    plot(tc2[i],color='k', linestyle=(0,(3,3)), label='Exposure 2',linewidth=1.5)
    remove_box()

    gca().set_xticks([0,2*np.pi])
    gca().set_xticklabels([])
    gca().set_xlim([0,2*np.pi])
    gca().set_yticks([0,round(tc1[i].max())+1])
    gca().set_yticklabels([0,round(tc1[i].max())+1],size=small)

    plt.annotate('C'+str(ct+1),(pi,round(tc1[i].max())+0.85),xycoords='data',size=xsmall)
    tickParams()
    if ct==1:
        legend(bbox_to_anchor=(1, 1.7), loc='upper right',ncol=2,frameon=False,columnspacing=1,handlelength=2,handletextpad=0.3,labelspacing=0.12)
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



gs9=fig.add_gridspec(1,1, left=0.57,right=0.69,bottom=0.35,top=0.47)
fig.add_subplot(gs9[0])
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_Exposure1_2_3strains.npy',allow_pickle=True)

x1=np.unwrap(dat[1])
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
plt.annotate('r = '+ str(round(p,2))+', p< 0.0001',(pi-0.3,0.5),size=xsmall)
remove_box()
tickParams()



##########################################################
# F. Bayesian Decoding
##########################################################

gs10=fig.add_gridspec(1,1, left=0.055,right=0.34,bottom=0.12,top=0.24)
fig.add_subplot(gs10[:])
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure1_decoding_exampleRD.npy',allow_pickle=True)
a1=dat[0]['actual']
a2=dat[0]['decode']
ep1= nts.IntervalSet(start=2.7415e+09, end=2.7415e+09+1.8e+8)#2.9715e+09) #dec_dat[0]['ep']

#light decoding
plot(nts.TsdFrame(a1).restrict(ep1).values, label='Actual', color='royalblue',alpha=0.9,linewidth=1.5)
plot(nts.TsdFrame(a2).restrict(ep1).values,label='Decoded',color='darkorange',alpha=0.8,linewidth=1.5)##DC3C95  good color for GNAT
gca().spines['left'].set_position(('axes',-0.01))

gca().set_xticks(np.linspace(0,900,4))
gca().set_xlim([0,900])
gca().set_xticklabels(['0','1','2','3'])
gca().set_xlabel('Time (min)',labelpad=0)

gca().set_ylim([0,2*np.pi])
gca().set_yticks([0,2*np.pi])
gca().set_yticklabels(['0\u00b0','360\u00b0'])
gca().set_ylabel('Head direction',labelpad=-4)
legend(bbox_to_anchor=(1.02, 1.28), loc='upper right',ncol=2,frameon=True,columnspacing=1,handlelength=2,handletextpad=0.3)
remove_box()
tickParams()
gca().tick_params(axis='y',which='major',pad=0.3, length=2,tickdir='out')
title('Mean Absolute Error= 20\u00b0 (12 HD cells)',loc='left',size=xsmall)




##########################################################
# G. Visual Cue Control ( rd1 vs WT)
##########################################################

gs11=fig.add_gridspec(2,2, left=0.40,right=0.52,bottom=0.12,top=0.24,hspace=0.6,wspace=0.6)
d=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_tcurve_cueRotLight_rd.npy',allow_pickle=True)
tc1=d[0]['tc_A']
tc2=d[0]['tc_B']

ids=[5,0]
ct=0
for i,x in enumerate(ids):
    ax=fig.add_subplot(gs11[i,0])
    remove_box()
    st=plot(tc1[x],c='k', label='standard')
    bl_n=plot(tc2[x],c='#907398',linestyle=(0,(3,3)),label='Blind: 90\u00b0'+'Visual cue rot.')
    gca().set_yticks([])
    gca().set_xticks([0,2*np.pi])
    gca().set_xticklabels([])
    gca().set_xlim([0,2*np.pi])
    tickParams()
    plt.annotate('C'+str(ct+1),(pi,round(tc2[x].max())+4),xycoords='data',size=xsmall)

    if ct ==0:
        gca().set_yticks([0,round(tc1[x].max())+5]) 
        gca().set_yticklabels([0,round(tc1[x].max())+5]) 
        title('rd1',y=1.35,size=small)

    else:
        gca().set_yticks([0,round(tc2[x].max())+5])
        gca().set_yticklabels([0,round(tc2[x].max())+5])
        gca().set_xticks([0, 2*pi])
        gca().set_xticklabels([0, 2*pi])
        gca().set_xticklabels(['0\u00b0','360\u00b0'])

    ct+=1
plt.ylabel('Firing rate (Hz)',y=1.2,labelpad=0)


dd=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_tcurve_cueRotLight_wt90deg.npy',allow_pickle=True).item()    
tc1=dd['tc_A'][0]
tc2=dd['tc_B'][0]

ids=[3,10]
ct=0
for i,x in enumerate(ids):
    fig.add_subplot(gs11[i,1])
    plot(tc1[x],c='k')
    wt_n=plot(tc2[x],c='#56B4E9',linestyle=(0,(3,3)),label='WT: 90\u00b0'+'Visual cue rot.')
    gca().set_yticks([])
    gca().set_xticks([0,2*np.pi])
    gca().set_xticklabels([])
    gca().set_xlim([0,2*np.pi])
    remove_box()
    tickParams()
    plt.annotate('C'+str(ct+1),(pi,round(tc2[x].max())+4),xycoords='data',size=xsmall)

    if ct ==0:
        gca().set_yticks([0,round(tc1[x].max())+5]) 
        gca().set_yticklabels([0,round(tc1[x].max())+5]) 
        title('WTL',y=1.35,size=small)

      
    else:
        gca().set_yticks([0,round(tc2[x].max())+5])
        gca().set_yticklabels([0,round(tc2[x].max())+5])
        gca().set_xticks([0, 2*pi])
        gca().set_xticklabels([0, 2*pi])
        gca().set_xticklabels(['0\u00b0','360\u00b0'])
    ct+=1
gca().set_xlabel('Head direction',x=-0.4,labelpad=0)
    
st_=Line2D([0],[0],color='k',linestyle='-',linewidth=1.5)
wt_=Line2D([0],[0],color='#56B4E9',linestyle=(0,(3,3)),linewidth=1.5)
bl_=Line2D([0],[0],color='#907398',linestyle=(0,(3,3)),linewidth=1.5)

labels=['90\u00b0 Cue rot.','90\u00b0 Cue rot.']
legend([bl_,wt_],labels,bbox_to_anchor=(1.2,3.2),frameon=False,labelspacing=0.12,handlelength=2,handletextpad=0.3,ncol=2)


### GAIN ###
gs12=fig.add_gridspec(1,1, left=0.56,right=0.68,bottom=0.12,top=0.21)
fig.add_subplot(gs12[0])

dat=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig1_cueRots3_Gain_ver3.npy',allow_pickle=True)

wt=dat[0]
blind=dat[1]
bins=np.linspace(0,1.2,20)

hist(blind,bins, histtype='stepfilled',density=True,edgecolor='k', color='#907398', alpha=0.85,linewidth=1,label='rd1')
hist(wt,bins, histtype='stepfilled',density=True, edgecolor='k',color='#56B4E9',linewidth=1,label='WTL',alpha=0.85)
remove_box()

gca().set_xticks([0,0.5,1])
gca().set_xticklabels(['0',0.5,1])

gca().set_yticks([0,3,6])
gca().set_yticklabels(['0','3','6'])


legend(loc='upper right',bbox_to_anchor=(1.15,1.05),frameon=False,handletextpad=0.3,ncol=1)
gca().set_ylabel('Normalized counts',labelpad=-0.2)

tickParams()
plt.annotate('p< 0.0001',(0,5.5),size=xsmall)
gca().set_xlabel('Gain (Visual cue control)',labelpad=0)

gs_dd3=fig.add_gridspec(3,2, left=0.73,right=0.83,bottom=0.12,top=0.47,hspace=0.4,wspace=1.3)




##########################################################
# H. Light vs Dark (rd1)
##########################################################
dat_dark1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_LightDark3_paired3Strains.npy',allow_pickle=True)  
rd_dark=dat_dark1[1]


fig.add_subplot(gs_dd3[0])
#Mean Firing Rate
expC='means'
x=rd_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=4)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_xticklabels([])
gca().set_ylim([0,29])
gca().set_yticks([0,14,28])
gca().set_yticklabels([0,14,28],size=small)


plot([0,1],[27,27],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.30,27.5),size=small)
gca().set_ylabel('Mean rate (Hz)',labelpad=-0.3)
remove_box()
tickParams()


fig.add_subplot(gs_dd3[1])
expC='peaks'
x=rd_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=4)
gca().set_xticks([0,1])
gca().set_xlim(-0.1,1.1)
gca().set_xticklabels([])
gca().set_yticks([0,40,80])
gca().set_yticklabels([0,40,80])

plot([0,1],[93,93],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.30,94),size=small)
gca().set_ylabel('Peak rate (Hz)',labelpad=-0.3)
remove_box()
tickParams()

fig.add_subplot(gs_dd3[2])
expC='mvl'
x=rd_dark[expC]
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
plt.annotate('n.s',(0.30,1.007),size=small)
gca().set_ylabel('Vector length (r)',size=small,labelpad=2)
remove_box()
tickParams()


fig.add_subplot(gs_dd3[3])
expC='info'
x=rd_dark[expC]
plt.plot([0,1],[x.light,x.dark],color='gray', alpha=0.5,linewidth=0.8)
plt.scatter([np.zeros(len(x)),np.ones(len(x))],[x.light,x.dark], s=1, color='k',zorder=3)
plt.plot([0,1],[mean(x.light),mean(x.dark)],color='red',linewidth=1.5,zorder=4)

gca().set_xticklabels(['Light','Dark'])
xticks(rotation=30)
gca().set_xlim(-0.1,1.1)
gca().set_xticks([0,1])
plot([0,1],[3.3,3.3],color='k',linewidth=1)
p=print(scipy.stats.wilcoxon(x.light,x.dark)[1])
plt.annotate('n.s',(0.30,3.35),size=small)
gca().set_ylabel('Info. (bits/spk)',labelpad=3)
gca().set_yticks([0,1,2,3]);gca().set_yticklabels(['0','1','2','3'])
remove_box()
tickParams()

fig.add_subplot(gs_dd3[4])
expC='width'
x=rd_dark[expC]
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
plt.annotate('n.s',(0.30,3.43),size=small)
gca().set_ylabel('Tuning width',labelpad=-2)
remove_box()
tickParams()
gca().tick_params(axis='y',which='both',pad=0.05, length=2)


#Panel Labels
plt.annotate('A',(0.025,0.76),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('B',(0.44,0.76),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('C',(0.73,0.76),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('D',(0.025,0.49),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('E',(0.37,0.49),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('F',(0.025,0.28),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('G',(0.37,0.28),xycoords='figure fraction',size=large,fontweight='bold')
plt.annotate('H',(0.696,0.49),xycoords='figure fraction',size=large,fontweight='bold')

# fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Figure1_py.pdf',dpi=600, format='pdf',transparent=True)


