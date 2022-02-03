# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 20:51:00 2022

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



################################################################################
#A. Polar plots
################################################################################

gsA=fig.add_gridspec(2,2,left=0.07,right=0.21,top=0.82,bottom=0.67)
tc=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\PAPER1\data\figures\figure2_example_tcurves_gnL2.npy', allow_pickle=True).item()

ep=tc['ep']
all_spks=tc['spk']
pos=tc['pos']
ids=[7,2,3,5]

#Polar plots
for i,x in enumerate(ids):
   fig.add_subplot(gsA[i], projection='polar')    
   plot(tc['wt_light'][x], label=str(x),linewidth=1.5,c='#DC3C95',zorder=6)
   gca().set_aspect('equal')
   hz=str(round(tc['wt_light'][x].max(),1))+'Hz'

   gca().fill(tc['wt_light'][x],'white',zorder=4)
        
   gca().xaxis.grid(False)
   gca().yaxis.grid(False)
   gca().set_yticklabels([])
   gca().set_xticklabels([])
        
   tick = [gca().get_rmax(), gca().get_rmax()*0.9]#0.9
   for t in np.deg2rad(np.arange(0, 360, 90)):
       gca().plot([t, t], tick, lw=1, color="gray")
    
   plt.annotate(hz,(38,33),xycoords='axes points', size=xsmall)
     


################################################################################
#B. ADN units characterised as HD cells
################################################################################

gsB1=fig.add_gridspec(1,1,left=0.07,right=0.21,top=0.59,bottom=0.54)
axB1=fig.add_subplot(gsB1[0])

plt.title('ADn units characterised as\nHD cells (%)',y=0.85,x=0.52,size=med)
plot([0.22,0.77],[0.4,0.4],linewidth=1,color='k')
plt.annotate('n.s',(0.45,0.435),size=small)
gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axB1.axis('off')
###############################################################################
gsB=fig.add_gridspec(1,2,left=0.07,right=0.21,top=0.565,bottom=0.49)


hd_d=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionDarkWT_ver3.npy',allow_pickle=True).item()
hd_l=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionLight3Strains_ver3.npy',allow_pickle=True)

a=hd_l[1]['hd_%']; a=list(map(lambda x: round(x,1),a))
b=hd_l[2]['hd_%']; b=list(map(lambda x: round(x,1),b))

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



#RD1
fig.add_subplot(gsB[0])
wedge,_=plt.pie([mean(a),100-mean(a)], colors=['white'],wedgeprops={"edgecolor":"#44AA99",'linewidth': 1.5,'antialiased': True},startangle=270)
wedge[1].set_visible(False)
gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

dat=str(cells[1][1])+'/'+str(cells[1][0]) +' cells'
plt.annotate(dat,(-0.9,-0.3),xycoords=gca().get_xaxis_transform(), size=small)
title(str(round(((cells[1][1])/(cells[1][0]))*100,1))+'%',y=0.46,size=small)
plt.annotate('(13 rd1 mice)',(1.5,-25),xycoords='axes points', size=small)


#GN
fig.add_subplot(gsB[1])
wedge,_=plt.pie([mean(b),100-mean(b)], colors=['white'],wedgeprops={"edgecolor":"#DC3C95",'linewidth': 1.5,'antialiased': True},startangle=270)
wedge[1].set_visible(False)
gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
stats=str(round(mean(b),1))+'%'#+' +/- '+str(round(std(a),1))+'%'

dat=str(cells[2][1])+'/'+str(cells[2][0])+ ' cells'
plt.annotate(dat,(-0.9,-0.3),xycoords=gca().get_xaxis_transform(), size=small)
title(str(round(((cells[2][1])/(cells[2][0]))*100,1))+'%',y=0.46,size=small)
plt.annotate('(8 animals)',(-1,-25),xycoords='axes points', size=small)



################################################################################
#C. Firing rate properties
################################################################################

gsC=fig.add_gridspec(4,1,left=0.3,right=0.43,top=0.92,bottom=0.30,hspace=0.35)

data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_LightV2_50z.npy',allow_pickle=True)#old
data1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_DarkV2_50z.npy',allow_pickle=True)


##############################################################################
###Mean Vector Length ##########################################################
con='vlength'

a=data[1][con].dropna().values.flatten()
b=data[2][con].dropna().values.flatten()


ax1=fig.add_subplot(gsC[0])

bns=np.linspace(0,1,10)
vp=ax1.violinplot([list(a),list(b)],widths=0.5,showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([a,b],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#44AA99','#DC3C95']
for i in range(2):
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

gsC1=fig.add_gridspec(1,1,left=0.32,right=0.41,top=0.95,bottom=0.90)
axC1=fig.add_subplot(gsC1[0])

plot([0.14,0.85],[0.4,0.4],linewidth=1,color='k')
plt.annotate('**',(0.45,0.31),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC1.axis('off')


##############################################################################
###Width ##########################################################
con='width'

a=data[1][con].dropna().values.flatten()
b=data[2][con].dropna().values.flatten()

ax2=fig.add_subplot(gsC[1])
bns=np.linspace(0,4,10)
cts=[histogram(i,bns,normed=True)[0].max() for i in [a,b]]
widths1=[(i/max(cts))*0.5 for i in cts]
widths2=[(len(a)/len(a))*0.5,(len(b)/len(a))*0.5]
widths=(((array(widths2)+array(widths1))/1)*0.5)+0.1

vp=ax2.violinplot([list(a),list(b)],widths=widths,showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([a,b],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#44AA99','#DC3C95']
for i in range(2):
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

gsC3=fig.add_gridspec(1,1,left=0.32,right=0.41,top=0.78,bottom=0.73)
axC3=fig.add_subplot(gsC3[0])

plot([0.14,0.85],[0.4,0.4],linewidth=1,color='k')
plt.annotate('**',(0.45,0.31),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC3.axis('off')



##############################################################################
###Mean Info ##########################################################
con='info'

a=data[1][con].dropna().values.flatten()
b=data[2][con].dropna().values.flatten()

ax4=fig.add_subplot(gsC[2])

bns=np.linspace(0,3,10)
cts=[histogram(i,bns,normed=True)[0].max() for i in [a,b]]
widths1=[(i/max(cts))*0.5 for i in cts]
widths2=[(len(a)/len(a))*0.5,(len(b)/len(a))*0.5]
widths=(((array(widths2)+array(widths1))/1)*0.5)+0.1

vp=ax4.violinplot([list(a),list(b)],widths=[0.55,0.55],showmeans=False, showmedians=False, showextrema=False,positions=[1,2])#widths=[0.14,0,0.82]
bp=boxplot([a,b],widths=0.13,showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#44AA99','#DC3C95']
for i in range(2):
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
gca().set_xticklabels(['rd1','Gnat1/2'])
remove_box()
tickParams()


# #PVALS 
gsC4=fig.add_gridspec(1,1,left=0.32,right=0.41,top=0.62,bottom=0.57)
axC4=fig.add_subplot(gsC4[0])

plot([0.14,0.85],[0.4,0.4],linewidth=1,color='k')
plt.annotate('*',(0.48,0.31),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axC4.axis('off')



# #Panel Labels
plt.annotate('A',(0.05,0.94),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('B',(0.05,0.62),xycoords='figure fraction',size=large, fontweight='bold')
plt.annotate('C',(0.25,0.94),xycoords='figure fraction',size=large, fontweight='bold')

# fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Figure3.pdf',dpi=600, format='pdf',transparent=True)
