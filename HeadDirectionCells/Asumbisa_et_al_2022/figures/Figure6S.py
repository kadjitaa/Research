# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:25:28 2022

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
    gca().tick_params(axis=axis,which='major',pad=2.5, length=2)
###############################################################################

plt.close('all')
fig=plt.figure(figsize=(11,8.5))
bkAx=bkgrid() #helper fxn to set grid in background


################################################################################
## Panel A1
gsA=fig.add_gridspec(2,1,left=0.04,right=0.24,top=0.92,bottom=0.63,hspace=0.1)
################################################################################

rings=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_allRings.npy',allow_pickle=True).item()

wt1,H1=rings['wt'][0][159]
wt2,H2=rings['wt'][2][165]

wt_rings=[wt1,wt2]
wt_Hs=[H1,H2]
size=25
for i in range(2):
    ax=fig.add_subplot(gsA[i])
    spk_iso=wt_rings[i]
    H=wt_Hs[i]
    gca().set_aspect(aspect=1)
    cmap=plt.cm.twilight((H-H.min())/(H.max()-H.min())) #H is a vector of the normalized actual position
    gca().scatter(spk_iso[:,0],spk_iso[:,1], c = cmap, marker = 'o', alpha = 0.7, linewidth = 1, s= size)
    gca().set_xticks([])
    gca().set_yticks([])
    sns.despine(ax=gca(),left=1,bottom=1)
    if i==0:
        ylabel('Control',size=med,labelpad=7)
        title('WT: Mouse #KA86',size=med,y=0.96)
    else:
        ylabel('Olfaction\nablated',size=med,labelpad=2)
        

################################################################################
## Panel A2
gsB=fig.add_gridspec(2,1,left=0.24,right=0.35,top=0.92,bottom=0.62,hspace=0)
################################################################################

rd1,H1=rings['rd1'][0][102]#102 vs 109
rd2,H2=rings['rd1'][1][106]

rd_rings=[rd1,rd2]
rd_Hs=[H1,H2]

for i in range(2):
    fig.add_subplot(gsB[i])
    spk_iso=rd_rings[i]
    H=rd_Hs[i]
    gca().set_aspect(aspect=1)
    cmap=plt.cm.twilight((H-H.min())/(H.max()-H.min())) #H is a vector of the normalized actual position
    gca().scatter(spk_iso[:,0],spk_iso[:,1], c = cmap, marker = 'o', alpha = 0.7, linewidth = 1, s= size)  
    gca().axis('off')
    if i==0:
        title('rd1: Mouse #KA65',size=med)
    else:
        pass



################################################################################
## Panel A3
gsC=fig.add_gridspec(2,1,left=0.39,right=0.5,top=0.92,bottom=0.62,hspace=0)
################################################################################

gn1,H1=rings['gnat'][0][131]#102 vs 109
gn2,H2=rings['gnat'][1][136]

gn_rings=[gn1,gn2]
gn_Hs=[H1,H2]

for i in range(2):
    fig.add_subplot(gsC[i])
    spk_iso=gn_rings[i]
    H=gn_Hs[i]
    gca().set_aspect(aspect=1)
    cmap=plt.cm.twilight((H-H.min())/(H.max()-H.min())) #H is a vector of the normalized actual position
    gca().scatter(spk_iso[:,0],spk_iso[:,1], c = cmap, marker = 'o', alpha = 0.7, linewidth = 1, s= size)  
    gca().axis('off')
    if i==0:
        title(r'Gnat1/2$^{mut}$'+': Mouse #KA79',size=med,y=0.95)
    else:
        pass


################################################################################
## Panel B
gsB11=fig.add_gridspec(2,1,left=0.62,right=0.95,top=0.92,bottom=0.67, hspace=0.5)
isoC=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig6_isoDecodingCurves.npy',allow_pickle=True).item()
################################################################################

a1=isoC['intact'][0]; d1=isoC['intact'][1]
a2=isoC['osn'][0]; d2=isoC['osn'][1]

fig.add_subplot(gsB11[0])
title('Control',size=med,loc='left')
plot(a1[350:650],color='#2d96e1',label="Actual",alpha=0.9,linewidth=1.5)
plot(d1[350:650],color='#d2691e',label="ISOMAP decoded",alpha=0.9,linewidth=1.5)
gca().set_ylim([0,2*np.pi])
gca().set_yticks([0,2*np.pi])
gca().set_yticklabels(['0\u00b0','360\u00b0'])
remove_box()
gca().set_xlim(0,300)
gca().set_xticks(np.arange(0,300+1,100))
gca().set_xticklabels([])
gca().set_ylabel('Head direction',labelpad=-4)
legend(bbox_to_anchor=(1.02, 1.28), loc='upper right',ncol=2,frameon=True,columnspacing=1,handlelength=2,handletextpad=0.3)
tickParams()
gca().tick_params(axis='x',which='major',pad=4)

fig.add_subplot(gsB11[1])
title('Olfaction ablated',size=med,loc='left')
plot(a2[340: 640],color='#2d96e1',alpha=0.9,linewidth=1.5)
plot(d2[340: 640],color='#d2691e',alpha=0.9,linewidth=1.5)
remove_box()
gca().set_ylim([0,2*np.pi])
gca().set_yticks([0,2*np.pi])
gca().set_yticklabels(['0\u00b0','360\u00b0'])
gca().set_xticks(np.arange(0,300+1,100))
gca().set_xticklabels(['0','20','40','60'])
gca().set_xlim(0,300)
gca().set_xlabel('Time (s)',labelpad=0)
gca().set_ylabel('Head direction',labelpad=-4)
tickParams()


################################################################################
## Panel C
gsB1=fig.add_gridspec(2,1,left=0.04,right=0.13,top=0.56,bottom=0.42)
gsB2=fig.add_gridspec(2,5,left=0.13,right=0.43,top=0.56,bottom=0.42)
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig6_TcurvDriftsBlindStable.npy',allow_pickle=True)
################################################################################

tc_avg=dat[0]
tcs_fang=dat[1]

for i in range(2):
    fig.add_subplot(gsB1[i,0],projection='polar')
    plot(tc_avg.iloc[:,-i+1],linewidth=1.5,c='k',zorder=6)
    gca().fill(tc_avg.iloc[:,-i+1],color='white',zorder=5)
    gca().set_aspect('equal')
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])
    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
    hz=str(round(tc_avg.iloc[:,-i+1].max(),1))+'Hz'
    plt.annotate(hz,(33,34),xycoords='axes points', size=xsmall)
    plt.annotate(str('C')+str(i+1),(-12,17),xycoords='axes points', size=small)
    if i==0:
        title('Full session',size=med,y=1.25)

for j in range(2):
    for i in range(5):
        fig.add_subplot(gsB2[j-1,i],projection='polar')
        plot(tcs_fang[i+1].iloc[:,j],linewidth=1.5,c='k',zorder=6)
        if j==1:
            gca().annotate(str(i+1), xy=(0.45, 1.12), xycoords='axes fraction',size=med)
        gca().fill(tcs_fang[i+1].iloc[:,j],color='white',zorder=5)
        gca().set_aspect('equal')
        gca().xaxis.grid(False)
        gca().yaxis.grid(False)
        gca().set_yticklabels([])
        gca().set_xticklabels([])
        tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
        for t in np.deg2rad(np.arange(0, 360, 90)):
            gca().plot([t, t], tick, lw=1, color="gray")   
        hz=str(round(tcs_fang[i+1].iloc[:,j].max(),1))+'Hz'
        plt.annotate(hz,(33,34),xycoords='axes points', size=xsmall)
        if j==1 and i==2:
            title('Short intervals',y=1.25,size=med)


##########################################################################################################
##Panel D
gsB3=fig.add_gridspec(1,1,left=0.49,right=0.59,top=0.57,bottom=0.42)
data1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig6s_mvlSTDeps.npy',allow_pickle=True).item()
data=pd.DataFrame(np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_mvlOSNeps.npy',allow_pickle=True))
##########################################################################################################

a=[]
for key,val in data1.items():
    a.extend(val['short_ep'])
b=data.loc[1].values

ax2=fig.add_subplot(gsB3[0])
vp=ax2.violinplot([a,b],showmeans=False, showmedians=False, showextrema=False,positions=[-0.3,1])#widths=[0.14,0,0.82]
bp=boxplot([a,b],positions=[-0.3,1],widths=0.15,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colors=['lightgray','lightgray']
for i in range(2):
   vp['bodies'][i].set_facecolor(colors[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.9)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
remove_box()
title('Short intervals',y=1.04)
plot([-0.3,1],[1,1],linewidth=1,color='k') 
plt.annotate('n.s',(0.2,1.02),size=small)
gca().set_ylabel('Vector length',labelpad=0)
gca().set_yticks([0,.5,1])
gca().set_yticklabels(['0','0.5','1'])
gca().set_xticklabels(['Control','No vision\nNo olfaction'])
tickParams()
gca().tick_params(axis='x',which='major',pad=4)



##########################################################################################################
##Panel E
gsD=fig.add_gridspec(1,1,left=0.63,right=0.95,top=0.57,bottom=0.42)
data1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftStatsLight.npy',allow_pickle=True)
data2=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftStats2.npy',allow_pickle=True)
###########################################################################################################

wtl=data1[0]['drift_light']; wtO=data2[0]['meanDrift']
rd=data1[1]['drift_light']; rdO=data2[1]['meanDrift']
gn=data1[2]['drift_light'];gnO=data2[2]['meanDrift']

fig.add_subplot(gsD[0])
bp=boxplot([wtl,wtO, rd,rdO,gn,gnO],positions=[-0.1,1.1,2.7,3.9,5.7,7.1],widths=0.4,showcaps=False,showfliers=False,zorder=5,patch_artist=True)
colors=['#56B4E9','k','#907398','k','#907398','k']
for i in range(6):
   bp['boxes'][i].set(facecolor=colors[i])
   bp['medians'][i].set(color='red')
   bp['medians'][i].set(linewidth=3) 
gca().set_xticks([-0.1,1.1,2.7,3.9,5.7,7.1]) 
labl='No vision\nNo olfaction'
gca().set_xticklabels([" ",'No vision\nNo olfaction','rd1','No olfaction'," ",'No olfaction'])  
plt.annotate(r'Gnat1/2$^{mut}$',(175,-12.5),xycoords='axes points', size=small)
plt.annotate(r'WT$_L$',(7,-12.5),xycoords='axes points', size=small)
remove_box()  

plot([-0.1,1.1],[1.4,1.4],linewidth=1,color='k') 
plt.annotate('**',(0.34,1.25),size=xlarge)
plot([2.7,3.9],[4.4,4.4],linewidth=1,color='k') 
plt.annotate('**',(3.15,4.23),size=xlarge)
plot([5.7,7.1],[3.5,3.5],linewidth=1,color='k') 
plt.annotate('**',(6.23,3.33),size=xlarge)
tickParams()
ylim(0,5)
ylabel('Mean PFD drift (\u00b0/s)',labelpad=1)
gca().tick_params(axis='x',which='major',pad=4)


#No Visio No Olfaction. drift fig must be split Split it 
##########################################################################################################
##Panel F----DDI by mouse line
gsE=fig.add_gridspec(3,1,left=0.07,right=0.23,top=0.34,bottom=0.07,hspace=0.5)
drift_dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftStats.npy',allow_pickle=True)
##########################################################################################################

a=drift_dat[0]['dci']
b=drift_dat[1]['dci']
c=drift_dat[2]['dci']

bins=np.linspace(-1,1,10)

labls=[r'WT$_D$','rd1',r'Gnat1/2$^{mut}$']
for i,x in enumerate([a,b,c]):
    fig.add_subplot(gsE[i+0])
    hist(x,bins= bins,histtype='stepfilled',density=True, color='gray',linewidth=2)
    gca().set_xticks([-1,0,1])
    gca().set_xticklabels(['-1','0','1'])
    gca().annotate(labls[i],(0.04,0.88),xycoords='axes fraction',size=small)
    if i==1:
        ylabel('Normalized counts',labelpad=1)
    if i==2:
        gca().set_xlabel('Drift direction index (DDI)')
    else:
        gca().set_xticklabels([])
    remove_box()
    tickParams()    
        
    
######################################################################################################################################
#Panel G
data4=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_DriftCurves.npy',allow_pickle=True)
###########################################################################################################
gsF=fig.add_gridspec(2,3,left=0.3,right=0.7,top=0.34,bottom=0.07,hspace=0.5,wspace=0.43)

anim=[91,92,93]; dft=['1.5'+'\u00b0/s' ,'2.4'+'\u00b0/s','4.6'+'\u00b0/s'] ; ddi=['0.91','0.92','0.79']


for i,j in enumerate(anim): 
    fig.add_subplot(gsF[1,i])
    x=data4[1][j][0]
    y=data4[1][j][1]
    ts=data4[1][j][2]
    plot(x,y,color='r')
    gca().set_xticks(np.linspace(ts[0],ts[-1],3))
    gca().set_xticklabels(['0','5','10'])
    if i==1:
        xlabel('Time (min)')
    gca().annotate('Day '+str(i+1),(0.04,0.885),xycoords='axes fraction',size=small)
    plt.annotate('Drift= '+str(dft[i])+'\nDDI= '+(ddi[i]),(41,15),xycoords='axes points', size=xsmall)
    remove_box()
    tickParams()
    if i==0:
        title('Mouse #KA63',loc='left')
        ylabel('PFD Drift (unwrap ang.)',labelpad=0.7,y=1.27)

     
anim1=[164,165]; dft=['0.4'+'\u00b0/s' ,'1.1'+'\u00b0/s'] ; ddi=['0.1','0.7']
for i,j in enumerate(anim1): 
    fig.add_subplot(gsF[0,i])
    x=data4[0][j][0]
    y=data4[0][j][1]
    ts=data4[0][j][2]
    plot(x,y,color='r')
    gca().set_xticks(np.linspace(ts[0],ts[-1],3))
    gca().set_xticklabels(['0','5','10'])
    gca().annotate('Day '+str(i+1),(0.04,0.885),xycoords='axes fraction',size=small)
    plt.annotate('Drift= '+str(dft[i])+'\nDDI= '+(ddi[i]),(41,15),xycoords='axes points', size=xsmall)
    remove_box()
    tickParams()
    if i==0:
        title('Mouse #KA86',loc='left')
        
        
        
    
######################################################################################################################################
#Panel H
data3=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_AHV_ADV.npy',allow_pickle=True)
data4=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig5_AHV_ADV_Curves.npy',allow_pickle=True)
dat=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig6_Shuffles.npy',allow_pickle=True)[0]

######################################################################################################################################
gsG=fig.add_gridspec(1,1,left=0.77,right=0.95,top=0.3,bottom=0.07,hspace=0.5,wspace=0.43)

fig.add_subplot(gsG[0])
 
hist(dat,100,color='darkgray')

shuffle_thres=np.percentile(dat,95) #positive
shff=vlines(shuffle_thres,0,2000,color='r',linestyle='dashed')
plt.annotate('95th perc.',(shuffle_thres,2006),color='r')
gca().set_ylabel('Counts',labelpad=1)
gca().set_xlabel('Pearson correlation (r)')
title('Shuffled distribution')
gca().set_yticks([0,500,1000,1500,2000])
gca().set_xlim(-1,1)
gca().set_xticks([-1,-0.5,0,0.5,1])
gca().set_xticklabels([-1,-0.5,'0',0.5,1])

remove_box()
tickParams()            
    


plt.annotate('A',(0.035,0.94),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.574,0.94),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.035,0.597),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('D',(0.455,0.597),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('E',(0.6,0.597),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('F',(0.035,0.35),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('G',(0.26,0.35),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('H',(0.72,0.35),xycoords='figure fraction',size=large, fontweight='bold')

bkAx.axis('off')
#fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/recent/Figure6S.pdf',dpi=600, format='pdf',transparent=True)
