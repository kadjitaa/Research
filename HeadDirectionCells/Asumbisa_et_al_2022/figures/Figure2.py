# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:56:30 2022

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
import scikit_posthocs as sp

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


#############################################
# A. Polar plots
#############################################
gsA=fig.add_gridspec(4,3,left=0.07,right=0.28,top=0.83,bottom=0.53)

tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_example_tcurves_wtL.npy', allow_pickle=True).item()

ep=tc['ep']
all_spks=tc['spk']
pos=tc['pos']
ids=[0,5,4,1]

#Polar plots
for i,x in enumerate(ids):
    fig.add_subplot(gsA[i,0], projection='polar')    
    plot(tc['wt_light'][x], label=str(x),linewidth=1.5,c='#56B4E9',zorder=5)
    remove_polarAx(gca(),True)
    gca().set_yticks([])
    gca().set_aspect('equal')
    hz=str(round(tc['wt_light'][x].max(),1))+'Hz'
    plt.annotate(hz,(36,33),xycoords='axes points', size=xsmall)
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().fill(tc['wt_light'][x],'white',zorder=4)

    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
        


tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_example_tcurves_rdL2.npy', allow_pickle=True).item()

ep=tc['ep']
all_spks=tc['spk']
pos=tc['pos']
ids=[5,8,9,1]

#Polar plots
for i,x in enumerate(ids):
    fig.add_subplot(gsA[i,1], projection='polar')    
    plot(tc['wt_light'][x], label=str(x),linewidth=1.5,c='#44AA99',zorder=5)
    gca().set_aspect('equal')
    hz=str(round(tc['wt_light'][x].max(),1))+'Hz'
    plt.annotate(hz,(36,33),xycoords='axes points', size=xsmall)
    gca().xaxis.grid(False)
    gca().yaxis.grid(False)
    gca().set_yticklabels([])
    gca().set_xticklabels([])

    gca().fill(tc['wt_light'][x],'white',zorder=4)

    tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
    for t in np.deg2rad(np.arange(0, 360, 90)):
        gca().plot([t, t], tick, lw=1, color="gray")
        

tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\Figure1\figure1_example_tcurves.npy',allow_pickle=True) .item()

ids=[18,17,19,14]
for i,x in enumerate(ids):
   fig.add_subplot(gsA[i,2], projection='polar')    

   plot(tc['dark'][x], label=str(x),linewidth=1.5,c='grey',zorder=5)
   gca().fill(tc['dark'][x],'white',zorder=4)
       
   gca().xaxis.grid(False)
   gca().yaxis.grid(False)
   gca().set_yticklabels([])
   gca().set_xticklabels([])
       
   tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
   for t in np.deg2rad(np.arange(0, 360, 90)):
       gca().plot([t, t], tick, lw=1, color="gray")
  
   hz=str(round(tc['dark'][x].max(),1))+'Hz'
   plt.annotate(hz,(36,33),xycoords='axes points', size=xsmall)
    




#############################################
# B. ADN units Characterised as HD cells
#############################################
gsB1=fig.add_gridspec(1,1,left=0.07,right=0.27,top=0.455,bottom=0.405)
axB1=fig.add_subplot(gsB1[0])

###Stats
#A vs ALL
plot([0.14,0.9],[0.4,0.4],linewidth=1,color='k')
plt.annotate('n.s',(0.30,0.43),size=small)
plt.annotate('**',(0.66,0.31),size=xlarge)

#B vs Remaining
plot([0.51,0.9],[0.22,0.22],linewidth=1,color='k')
plt.annotate('**',(0.66,0.13),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axB1.axis('off')
plt.title('ADn units characterised as HD cells (%)',y=1,x=0.52,size=med)


gsB=fig.add_gridspec(1,4,left=0.07,right=0.35,top=0.42,bottom=0.34)
hd_d=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionDarkWT_ver3.npy',allow_pickle=True).item()
hd_l=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionLight3Strains_ver3.npy',allow_pickle=True)

a=hd_l[0]['hd_%']; a=list(map(lambda x: round(x,1),a))
b=hd_l[1]['hd_%']; b=list(map(lambda x: round(x,1),b))
d=hd_d['hd_%']; d=list(map(lambda x: round(x,1),d))


cells={}
for i in range(3):
    tot=0;hd=0
    for key, val in hd_l[i]['counts'].items():
        tot+=val[0]
        hd+=val[1]
    
    cells[i]=[tot,hd]


totd=0;hdd=0
for key, val in hd_d['counts'].items():
    totd+=val[0]
    hdd+=val[1]


#WTl
fig.add_subplot(gsB[0])
wedge,_=plt.pie([mean(a),100-mean(a)], colors=['white'],wedgeprops={"edgecolor":"#56B4E9",'linewidth': 1.5,'antialiased': True},startangle=270)
wedge[1].set_visible(False)
gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
dat=str(cells[0][1])+'/'+str(cells[0][0])
plt.annotate(dat,(-0.9,-0.3),xycoords=gca().get_xaxis_transform(), size=small)
title(str(round(((cells[0][1])/(cells[0][0]))*100,1))+'%',y=0.46,size=small)
plt.annotate('(9 animals)',(1.5,-25),xycoords='axes points', size=small)



#RD1
fig.add_subplot(gsB[1])
wedge,_=plt.pie([mean(b),100-mean(b)], colors=['white'],wedgeprops={"edgecolor":"#44AA99",'linewidth': 1.5,'antialiased': True},startangle=270)
wedge[1].set_visible(False)
gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
stats=str(round(mean(b),1))+'%'#+' +/- '+str(round(std(a),1))+'%'

dat=str(cells[1][1])+'/'+str(cells[1][0])
plt.annotate(dat,(-0.9,-0.3),xycoords=gca().get_xaxis_transform(), size=small)
title(str(round(((cells[1][1])/(cells[1][0]))*100,1))+'%',y=0.46,size=small)
plt.annotate('(13 animals)',(-1,-25),xycoords='axes points', size=small)


#WTd
fig.add_subplot(gsB[2])
wedge,_=plt.pie([mean(d),100-mean(d)], colors=['white'],wedgeprops={"edgecolor":"grey",'linewidth': 1.5,'antialiased': True},startangle=290)
wedge[1].set_visible(False)
gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
stats=str(round(mean(d),1))+'%'#+' +/- '+str(round(std(a),1))+'%'

dat=str(hdd)+'/'+str(totd)
plt.annotate(dat,(-0.9,-0.3),xycoords=gca().get_xaxis_transform(), size=small)
title(str(round(((hdd)/(totd))*100,1))+'%',y=0.46,size=small)
plt.annotate('(6 animals)',(1.5,-25),xycoords='axes points', size=small)



#############################################
# C. Firing Properties
#############################################
gsC=fig.add_gridspec(4,1,left=0.37,right=0.55,top=0.92,bottom=0.30,hspace=0.35)

data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_LightV2_50z.npy',allow_pickle=True)#old
data1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_DarkV2_50z.npy',allow_pickle=True)

##############################################################################
###Mean Vector Length ##########################################################
con='vlength'

a=data[0][con].dropna().values.flatten()  #WT-Light
b=data[1][con].dropna().values.flatten()  #rd1
d=data1[0][con].dropna().values.flatten() #WT-dark


ax1=fig.add_subplot(gsC[0])
bns=np.linspace(0,1,10)
vp=ax1.violinplot([list(a),list(b),list(d)],widths=0.6,showmeans=False, showmedians=False, showextrema=False,positions=[1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,d],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','darkgray']
for i in range(3):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

remove_box()
gca().set_ylabel('Vector length (r)')
gca().set_xticklabels([])

tickParams()
gca().set_yticks([0,.5,1])
gca().set_yticklabels([0,.5,1])



gsC1=fig.add_gridspec(1,1,left=0.37,right=0.55,top=0.95,bottom=0.9)
axC1=fig.add_subplot(gsC1[0])

plot([0.165,0.825],[0.4,0.4],linewidth=1,color='k')
plt.annotate('*',(0.315,0.31),size=14)
plt.annotate('**',(0.63,0.31),size=xlarge)

# #B vs Remaining
plot([0.495,0.825],[0.22,0.22],linewidth=1,color='k')
plt.annotate('**',(0.63,0.13),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC1.axis('off')



#############################################################################
###Mean Stability(r) ##########################################################
con='corr'

a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax3=fig.add_subplot(gsC[1,0])
bns=np.linspace(-1,1,20)

vp=ax3.violinplot([list(a),list(b),list(d)],widths=0.6,showmeans=False, showmedians=False, showextrema=False,positions=[1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,d],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','darkgray']
for i in range(3):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

remove_box()

gca().set_ylabel('Stability (r)')
gca().set_xticklabels([])
gca().set_ylim([0,1.1])
gca().set_yticks([0,0.5,1])
gca().set_yticklabels([0,0.5,1])
tickParams()


############################################

gsC2=fig.add_gridspec(1,1,left=0.37,right=0.55,top=0.79,bottom=0.74)
axC2=fig.add_subplot(gsC2[0])

plot([0.165,0.825],[0.4,0.4],linewidth=1,color='k')
plt.annotate('*',(0.315,0.31),size=14)
plt.annotate('**',(0.63,0.31),size=xlarge)
plot([0.495,0.825],[0.22,0.22],linewidth=1,color='k')
plt.annotate('**',(0.63,0.13),size=xlarge)
gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC2.axis('off')


##############################################################################
###Width ##########################################################
con='width'

a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax2=fig.add_subplot(gsC[2,0])
bns=np.linspace(0,4,10)

vp=ax2.violinplot([list(a),list(b),list(d)],widths=0.6,showmeans=False, showmedians=False, showextrema=False,positions=[1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,d],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','darkgray']
for i in range(3):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

remove_box()
gca().set_yticks([deg2rad(45),deg2rad(135),deg2rad(225)])
gca().set_yticklabels(['45\u00b0','135\u00b0','225\u00b0'])
gca().set_ylabel('Tuning width',labelpad=1)
gca().set_xticklabels([])
tickParams()


gsC3=fig.add_gridspec(1,1,left=0.37,right=0.55,top=0.615,bottom=0.565)
axC3=fig.add_subplot(gsC3[0])

plot([0.165,0.825],[0.4,0.4],linewidth=1,color='k')
plt.annotate('**',(0.30,0.31),size=14)
plt.annotate('**',(0.63,0.31),size=xlarge)

# #B vs Remaining
plot([0.495,0.825],[0.22,0.22],linewidth=1,color='k')
plt.annotate('**',(0.63,0.13),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC3.axis('off')




##############################################################################
###Mean Info ##########################################################
con='info'


a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax4=fig.add_subplot(gsC[3,0])

bns=np.linspace(0,3,10)

vp=ax4.violinplot([list(a),list(b),list(d)],widths=0.6,showmeans=False, showmedians=False, showextrema=False,positions=[1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,d],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','darkgray']
for i in range(3):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


gca().set_yticks([0,1.5,3])
gca().set_yticklabels(['0',1.5,3])
ylim(0,3)

gca().set_ylabel('Info. (bits/spk)')
gca().set_xticklabels([r'WT$_L$','rd1',r'WT$_D$'])
remove_box()
tickParams()


# #PVALS 

gsC4=fig.add_gridspec(1,1,left=0.37,right=0.55,top=0.45,bottom=0.40)
axC4=fig.add_subplot(gsC4[0])

plot([0.165,0.825],[0.4,0.4],linewidth=1,color='k')
plt.annotate('**',(0.30,0.31),size=14)
plt.annotate('**',(0.63,0.31),size=xlarge)
plot([0.495,0.825],[0.22,0.22],linewidth=1,color='k')
plt.annotate('**',(0.63,0.13),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC4.axis('off')



#Panel Labels
plt.annotate('A',(0.05,0.95),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.05,0.48),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.32,0.95),xycoords='figure fraction',size=large, fontweight='bold')


# fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Fig2_1_4_py.pdf',dpi=600, format='pdf',transparent=True)
