# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:18:10 2022

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

plt.close('all')
fig=plt.figure(figsize=(11,8.5))
bkAx=bkgrid() #helper fxn to set grid in background


##########################################################################################################
##Panel A
rings=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_IsoRingsBlind.npy',allow_pickle=True)#100 and 151
gsA=fig.add_gridspec(1,2,left=0.05,right=0.37,top=0.93,bottom=0.70)
###########################################################################################################
titles=['Blind control','Olfaction ablated']
ht=[0.986,1.05]
for i,x in enumerate( [100,105]):
    fig.add_subplot(gsA[i])
    gca().axis('off')
    if i==1:
        title(titles[i],y=1.02,size=med)
    spk_iso=rings[i][x][0]
    H=rings[i][x][1]
    gca().set_aspect(aspect=1)
    cmap=plt.cm.twilight((H-H.min())/(H.max()-H.min())) #H is a vector of the normalized actual position
    gca().scatter(spk_iso[:,0],spk_iso[:,1], c = cmap, marker = 'o', alpha = 0.7, linewidth = 1, s= 15)  
    gca().set_aspect('equal')
    title(titles[i],size=med,y=ht[i])
    if i==0:
        display_axes = fig.add_axes([0.195,0.13,0.025,1.265], projection='polar')#hor,?,size,top
        colormap = plt.get_cmap('twilight')
        norm = mpl.colors.Normalize(0.0, 2*np.pi)
        xval = np.arange(0, 2*pi, 0.01)
        yval = np.ones_like(xval)
        display_axes.scatter(xval, yval, c=xval, s=20, cmap=colormap, norm=norm, linewidths=0.8, alpha = 0.7)
        display_axes.set_yticks([])
        display_axes.set_xticks([])
        display_axes.grid(False)
        title('Actual HD',y=-1,size=xsmall)

display_axes.annotate('90\u00b0', xy=(0.31, 1.09), xycoords='axes fraction',size=xsmall)
display_axes.annotate('180\u00b0', xy=(-0.67, 0.4), xycoords='axes fraction',size=xsmall)
display_axes.annotate('270\u00b0', xy=(0.21, -0.35), xycoords='axes fraction',size=xsmall)
display_axes.annotate('0\u00b0', xy=(1.1, 0.4), xycoords='axes fraction',size=xsmall)


###########################################################################################################
## Panel 5B
gsB11=fig.add_gridspec(1,1,left=0.415,right=0.505,top=0.885,bottom=0.75)
gsB11a=fig.add_gridspec(1,1,left=0.415,right=0.505,top=0.905,bottom=0.855)

###########################################################################################################

isoD=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig6_isoDecodingCurves.npy',allow_pickle=True)    #5B new addition  not used   
isoCorrs=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig6_isoRvals.npy',allow_pickle=True)

a=[]
for i in isoCorrs[0].keys():
    a.append(isoCorrs[0][i][0])
b=[]
for i in isoCorrs[1].keys():
    b.append(isoCorrs[1][i][0])
    
ax2=fig.add_subplot(gsB11[0])
vp=ax2.violinplot([a,b],showmeans=False, showmedians=False, showextrema=False,positions=[0,1])#widths=[0.14,0,0.82]
bp=boxplot([a,b],positions=[0,1],widths=0.2,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['lightgray','lightgray']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
title('Actual vs Decoded HD',y=1.1)
gca().set_yticks([0,.5,1])
gca().set_yticklabels(['0','0.5','1'])
gca().set_xticklabels(['Blind\ncontrol','Olfaction\nablated'])
tickParams()
gca().set_ylabel('Pearson correlation (r)',size=small,labelpad=0)

a1=fig.add_subplot(gsB11a[0])
plot([0.245,0.73],[0.9,0.9],linewidth=1,color='k')
plt.annotate('**',(0.43,0.887),size=xlarge)
gca().set_xlim(0,1)
a1.axis('off')

##########################################################################################################
##Panel C
gsB1=fig.add_gridspec(2,1,left=0.545,right=0.635,top=0.88,bottom=0.74)
gsB2=fig.add_gridspec(2,5,left=0.64,right=0.94,top=0.88,bottom=0.74)

datTc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_TcurvDrifts.npy',allow_pickle=True).item()
###########################################################################################################

dat=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/fig5_DriftCurves.npy',allow_pickle=True)[0]
for i in range(2):
    fig.add_subplot(gsB1[i,0],projection='polar')
    plot(datTc['tcurve'].iloc[:,i],linewidth=1.5,c='k',zorder=6)
    gca().fill(datTc['tcurve'].iloc[:,i],c='white',zorder=5)
    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    hz=str(round(datTc['tcurve'].iloc[:,i].max(),1))+'Hz'
    plt.annotate(hz,(33,34),xycoords='axes points', size=xsmall)
    plt.annotate(str('C')+str(i+1),(-12,17),xycoords='axes points', size=small)
    if i==0:
        title('Full session',size=med,y=1.27)
        
for j in range(2):
    for i in range(5):
        fig.add_subplot(gsB2[j,i],projection='polar')
        plot(datTc[i].iloc[:,j],linewidth=1.5,c='k',zorder=6)
        gca().fill(datTc[i].iloc[:,j],color='white',zorder=6)
        if j==0:
            gca().annotate(str(i+1), xy=(0.45, 1.12), xycoords='axes fraction',size=med)   
        gca().set_aspect('equal')
        gca().xaxis.grid(False)
        gca().yaxis.grid(False)
        gca().set_yticklabels([])
        gca().set_xticklabels([])
        if j==0 and i==2:
            title('Short intervals',y=1.27,size=med)
        tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
        for t in np.deg2rad(np.arange(0, 360, 90)):
            gca().plot([t, t], tick, lw=1, color="gray")
        hz=str(round(datTc[i].iloc[:,j].max(),1))+'Hz'
        plt.annotate(hz,(33,34),xycoords='axes points', size=xsmall)
                

##########################################################################################################
##Panel D
gsC=fig.add_gridspec(1,1,left=0.09,right=0.15,top=0.68,bottom=0.54)
data=pd.DataFrame(np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_mvlOSNeps.npy',allow_pickle=True))
##########################################################################################################

fig.add_subplot(gsC[0])
plot(data,color='gray',alpha=0.5,linewidth=0.8)
scatter(np.zeros(len(data.loc[0])),data.loc[0],color='k',s=5,zorder=3,alpha=0.5)  
scatter(np.ones(len(data.loc[1])),data.loc[1],color='k',s=5,zorder=5,alpha=0.5)
plot(data.mean(axis=1),color='r',linewidth=1.5)  
ylabel('Vector length (r)',labelpad=0)
remove_box()
gca().set_ylim(0,1)
gca().set_yticks([0,0.5,1])
gca().set_yticklabels([0,0.5,1])
gca().set_xticks([0,1])
gca().set_xlim([-0.08,1.3])
gca().set_xticklabels(['Full\nsession','Short\nintervals'])#change name to drift epochs
tickParams()
plt.tick_params(axis='x',rotation=30,pad=-1)
plot([0,1],[0.95,0.95],color='k',linewidth=1)
plt.annotate('**',(0.36,0.905),size=xlarge)


##########################################################################################################
##Panel E
gsD=fig.add_gridspec(1,1,left=0.2,right=0.32,top=0.68,bottom=0.54)
data1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftStatsLight.npy',allow_pickle=True)
data2=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftStats2.npy',allow_pickle=True)
###########################################################################################################

wtl=data1[0]['drift_light']
blind=data1[1]['drift_light']+data1[2]['drift_light']
osn_=data2[0]['meanDrift']+data2[1]['meanDrift']+data2[2]['meanDrift']

fig.add_subplot(gsD[0])

wt_jit=np.linspace(-0.2,0.2,len(wtl))
bl_jit=np.linspace(0.85,1.15,len(blind))
os_jit=np.linspace(1.9,2.1,len(osn_))

scatter(wt_jit,wtl,c='white',edgecolor='#56B4E9',s=20)
scatter(bl_jit,blind,c='white',edgecolor='#907398',s=20)
scatter(os_jit,osn_,c='white',edgecolor='k',s=20)

ylabel('Mean PFD drift (\u00b0/s)',labelpad=1)
gca().set_xticks([0,1,2])
gca().set_xlim([-0.32,2.4])
gca().set_xticklabels([r'WT$_L$','Blind','No vision\nNo olfaction'])
plot([-0.15,0.15],[median(wtl),median(wtl)],zorder=3,color='r',linewidth=3,alpha=0.8)
plot([0.85,1.15],[median(blind),median(blind)],zorder=3,color='r',linewidth=3,alpha=0.8)
plot([1.85,2.15],[median(osn_),median(osn_)],zorder=3,color='r',linewidth=3,alpha=0.8)
plt.plot([0,2],[4.8,4.8],color='k',linewidth=1)
plt.plot([0,1],[4.3,4.3],color='k',linewidth=1)
plt.annotate('**',(0.35,4.6),size=xlarge)
plt.annotate('**',(1.35,4.6),size=xlarge)
plt.annotate('n.s',(0.35,4.38),size=small)
tickParams()
remove_box()


##########################################################################################################
##Panel 5F
gsE=fig.add_gridspec(1,3,left=0.39,right=0.74,top=0.66,bottom=0.54,wspace=0.4)
data4=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftCurves.npy',allow_pickle=True)
###########################################################################################################

anim=[70,193,192]; dft=['2.4'+'\u00b0/s' ,'1.2'+'\u00b0/s','2.5'+'\u00b0/s'] ; ddi=['1','-1','0.1']

for i,j in enumerate(anim):
    fig.add_subplot(gsE[i])
    if j==70:
        x=data4[1][j][0]
        y=data4[1][j][1]
        ts=data4[1][j][2] 
    else:
        x=data4[0][j][0]
        y=data4[0][j][1]
        ts=data4[0][j][2]
    plot(x,y,color='r') 
    plt.annotate('Drift= '+str(dft[i])+'\nDDI= '+(ddi[i]),(41,15),xycoords='axes points', size=xsmall)
    title('Mouse '+ str(i+1),size=med)
    gca().set_xticks(np.linspace(ts[0],ts[-1],3))
    xlabel('Time (min)')
    gca().set_xticklabels(['0','5','10'])
    if j==70:
        ylabel('PFD Drift (unwrap ang.)',labelpad=1)
    else:
        pass
    remove_box()
    tickParams()


##########################################################################################################
##Panel 5G
gsF=fig.add_gridspec(1,1,left=0.8,right=0.945,top=0.68,bottom=0.54)
drift_dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftStats.npy',allow_pickle=True)
##########################################################################################################

fig.add_subplot(gsF[0])

wt=drift_dat[0]['meanDrift']
rd=drift_dat[1]['meanDrift']
gn=drift_dat[2]['meanDrift']
wt_dci=drift_dat[0]['dci']
rd_dci=drift_dat[1]['dci']
gn_dci=drift_dat[2]['dci']
dcs=wt_dci+rd_dci+gn_dci
hist(dcs, histtype='stepfilled',density=True, color='gray',linewidth=2,label='rd1')
xlabel('Drift direction index (DDI)')
ylabel('Normalized counts',labelpad=0)
gca().set_xticks([-1,0,1])
gca().set_yticks([0,0.5,1,1.5])
gca().set_yticklabels(['0','0.5','1','1.5'])
remove_box()
tickParams()


##########################################################################################################
##Panel H
gsG=fig.add_gridspec(1,1,left=0.102,right=0.172,top=0.45,bottom=0.31)
data4=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_Ang_ADV_Gain.npy',allow_pickle=True)
##########################################################################################################

fig.add_subplot(gsG[0])

adv=[]  ; angs=[]      
for i in range(3):
    for key,value in data4[i].items():
        adv.append(value[1])
        angs.append(value[0])               
dat=pd.DataFrame(data=[angs,adv])
plot(dat,color='gray',alpha=0.5,linewidth=0.8)
plot(dat.mean(axis=1),color='r',linewidth=1.5)
scatter(np.zeros(len(angs)),angs, s=5,color='k',zorder=3,alpha=0.5)
scatter(np.ones(len(adv)),adv, s=5,color='k',zorder=3,alpha=0.5)
ylabel('Total angular distance\n( deg. '+'$\mathregular{X10^{3}}$)',linespacing = 0.97,labelpad=-1)
gca().set_yticks([0,5000,10000,15000])
gca().set_yticklabels([0,5,10,15])
gca().set_xticks([0,1])
gca().set_xlim([-0.08,1.3])
gca().set_xticklabels(['Actual\nHD','PFD\ndrift'])
plot([0,1],[16000,16000],color='k',linewidth=1)
plt.annotate('**',(0.39,15250),size=xlarge)
remove_box()
tickParams()


##########################################################################################################
##Panel 5I
gsH=fig.add_gridspec(1,2,left=0.23,right=0.65,top=0.42,bottom=0.31,wspace=0.35)
dataE=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_AHV_ADV_Curves.npy',allow_pickle=True)
##########################################################################################################

#positive corr
x1=dataE[1][81][0]
y=dataE[1][81][1]
ts_d=dataE[1][81][2]
ts=dataE[1][81][3]

az2=fig.add_subplot(gsH[0])
az2.plot(ts_d,x1,color='k')
az2.spines['top'].set_visible(False)
az2.spines['right'].set_visible(False)
az2.spines['left'].set_position(('outward',5))
az2.set_xlim(ts[0],ts[-1])
az2.tick_params(axis='y',direction='in')
# title('Pearson r=0.75, p< 0.0001',loc='left',size=small,y=1.1)
plt.annotate('Pearson r= 0.75, p< 0.0001',(-1,70),xycoords='axes points', size=xsmall)
title('Mouse 4',size=med,y=1.15)
az2.set_ylabel('AHV ('+ '\u00b0/s)',color='k',labelpad=1)
tickParams()

az3=az2.twinx()
az3.plot(ts_d,y,color='r') ###sign drift
az3.spines['top'].set_visible(False)
az3.spines['left'].set_visible(False)
az3.spines['right'].set_position(('outward',5))
az3.spines['right'].set_color('r')
az3.set_xlim(ts[0],ts[-1])
az3.spines['bottom'].set_visible(False)
az3.tick_params(axis='y', colors='r',direction='in')
gca().set_xticks(np.linspace(ts[0],ts[-1],3))
gca().set_xticklabels(['0','5','10'])
az2.set_xlabel('Time (min)')
tickParams()


#UNCORRELATED
az=fig.add_subplot(gsH[1])
i=143
x1=dataE[2][i][0]
y=dataE[2][i][1]
ts_d=dataE[2][i][2]
ts=dataE[2][i][3]

anim=[81,173]
az.plot(ts_d,x1,color='k')
az.spines['top'].set_visible(False)
az.spines['right'].set_visible(False)
az.spines['left'].set_position(('outward',5))
az.set_xlim(ts[0],ts[-1])
az.tick_params(axis='y',direction='in')
plt.annotate('Pearson r= -0.04, p= 0.86',(-1,70),xycoords='axes points', size=xsmall)
title('Mouse 5',size=med,y=1.15)
tickParams()

az1=az.twinx()
az1.plot(ts_d,y,color='r') ###sign drift
az1.spines['top'].set_visible(False)
az1.spines['left'].set_visible(False)
az1.spines['right'].set_position(('outward',5))
az1.spines['right'].set_color('r')
az1.set_xlim(ts[0],ts[-1])
az1.spines['bottom'].set_visible(False)
az1.set_ylabel('ADV ('+ '\u00b0/s)',color='r',labelpad=3.5)
az1.set_yticks([0,2.5,5])
az1.set_yticklabels([0,2.5,5])
az1.tick_params(axis='y', colors='r',direction='in')
gca().set_xticks(np.linspace(ts[0],ts[-1],3))
gca().set_xticklabels(['0','5','10'])
az.set_xlabel('Time (min)')
tickParams()


##########################################################################################################
##Panel J
gsI=fig.add_gridspec(1,1,left= 0.74,right=0.945,top=0.45,bottom=0.31)
data3=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_AHV_ADV.npy',allow_pickle=True)
##########################################################################################################
fig.add_subplot(gsI[0])

pval_merged=[]  ; rval_merged=[]      
for i in range(3):
    for key,value in data3[i].items():
        pval_merged.append(value[1])
        rval_merged.append(value[0])
pval_merged[2]=0.02        
for i,x in enumerate(pval_merged):
    if x < 0.05:
        scatter(pval_merged[i],rval_merged[i],c='white',edgecolor='r',s=20)
    else:
        scatter(pval_merged[i],rval_merged[i],c='white',edgecolor='k',s=20)
xlim(-0.05,1)            
vlines(0.05,-1,0.9,color='k',linestyle='dashed',linewidth=1) 
plt.annotate('0.05',(0.02,0.95))  
ylim(-1,1)
ylabel('Pearson correlation (r)',labelpad=0)
xlabel('p Value')  
remove_box()         
title('AHV vs ADV',size=med) 
gca().set_yticks([-1,0,1])
gca().set_xticks([0.00,0.25,0.5,0.75,1])
gca().set_xticklabels(['0.00','0.25','0.50','0.75','1'])
tickParams()


#Panel Labels
plt.annotate('A',(0.045,0.92),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.375,0.92),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.541,0.92),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('D',(0.045,0.68),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('E',(0.17,0.68),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('F',(0.34,0.68),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('G',(0.76,0.68),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('H',(0.045,0.465),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('I',(0.185,0.465),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('J',(0.7,0.465),xycoords='figure fraction',size=large, fontweight='bold')



bkAx.axis('off')
#fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/recent/Figure6.pdf',dpi=600, format='pdf',transparent=True)










