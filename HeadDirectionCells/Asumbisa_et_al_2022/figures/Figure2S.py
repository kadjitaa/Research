# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 20:32:41 2022

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


################################################################################
#A. Firing rate properties
################################################################################

gsB1=fig.add_gridspec(1,1,left=0.07,right=0.35,top=0.9,bottom=0.84)
axB1=fig.add_subplot(gsB1[0])

###Stats
#A vs ALL
plot([0.1,0.89],[0.5,0.5],linewidth=1,color='k')
plt.annotate('n.s',(0.22,0.53),size=small)
plt.annotate('n.s',(0.48,0.53),size=small)
plt.annotate('**',(0.74,0.44),size=xlarge)

#B vs Remaining
plot([0.37,0.89],[0.3,0.3],linewidth=1,color='k')
plt.annotate('**',(0.74,0.25),size=xlarge)
plt.annotate('n.s',(0.48,0.34),size=small)

#Cvs D
plot([0.63,0.89],[0.1,0.1],linewidth=1,color='k')
plt.annotate('**',(0.74,0.04),size=xlarge)

gca().set_ylim(0,0.7)
gca().set_xlim(0,1)
axB1.axis('off')
plt.title('ADn units characterised as HD cells (%)',y=1,size=med)
###############################################################################

gsB=fig.add_gridspec(1,4,left=0.07,right=0.35,top=0.85,bottom=0.78)

hd_d=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionDarkWT_ver3.npy',allow_pickle=True).item()
hd_l=np.load(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/paper1_dataset/fig1_hdsDistributionLight3Strains_ver3.npy',allow_pickle=True)

a=hd_l[0]['hd_%']; a=list(map(lambda x: round(x,1),a))
b=hd_l[1]['hd_%']; b=list(map(lambda x: round(x,1),b))
c=hd_l[2]['hd_%']; b=list(map(lambda x: round(x,1),b))

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

#GNAT
fig.add_subplot(gsB[2])
wedge,_=plt.pie([mean(c),100-mean(c)], colors=['white'],wedgeprops={"edgecolor":"#DC3C95",'linewidth': 1.5,'antialiased': True},startangle=270)
wedge[1].set_visible(False)
gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
stats=str(round(mean(c),1))+'%'#+' +/- '+str(round(std(a),1))+'%'
dat=str(cells[2][1])+'/'+str(cells[2][0])
plt.annotate(dat,(-0.9,-0.3),xycoords=gca().get_xaxis_transform(), size=small)
title(str(round(((cells[2][1])/(cells[2][0]))*100,1))+'%',y=0.46,size=small)
plt.annotate('(8 animals)',(1.5,-25),xycoords='axes points', size=small)

#WTd
fig.add_subplot(gsB[3])
wedge,_=plt.pie([mean(d),100-mean(d)], colors=['white'],wedgeprops={"edgecolor":"darkgray",'linewidth': 1.5,'antialiased': True},startangle=290)
wedge[1].set_visible(False)
gca().set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
stats=str(round(mean(d),1))+'%'#+' +/- '+str(round(std(a),1))+'%'
dat=str(hdd)+'/'+str(totd)
plt.annotate(dat,(-0.9,-0.3),xycoords=gca().get_xaxis_transform(), size=small)
title(str(round(((hdd)/(totd))*100,1))+'%',y=0.46,size=small)
plt.annotate('(6 animals)',(1.5,-25),xycoords='axes points', size=small)




################################################################################
#B. Firing rate properties
################################################################################

gsC=fig.add_gridspec(3,2,left=0.05,right=0.53,top=0.68,bottom=0.14,hspace=0.6,wspace=0.3)

data=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_LightV2_50z.npy',allow_pickle=True)
data1=np.load(r'C:\Users\kasum\Dropbox\ADn_Project\ADn_Figs\fig1_DarkV2_50z.npy',allow_pickle=True)


con='means'
a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
c=data[2][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax1=fig.add_subplot(gsC[0])
vp=ax1.violinplot([list(a),list(b),list(c),list(d)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,c,d],widths=0.13,positions=[0,1,2,3],showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','#DC3C95','darkgray']
for i in range(4):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

remove_box()
gca().set_ylabel('Mean rate (Hz)')
tickParams()
gca().set_xticklabels([])
gca().set_yticks([0,10,20,30,40])
ylim(0,42)

##############################################################################
###PEAK FIRING RATE ##########################################################
con='peaks'
a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
c=data[2][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax2=fig.add_subplot(gsC[1])

vp=ax2.violinplot([list(a),list(b),list(c),list(d)],widths=0.7,showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,c,d],widths=0.13,positions=[0,1,2,3],showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','#DC3C95','darkgray']
for i in range(4):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

gca().set_ylabel('Peak rate (Hz)',labelpad=-1)
ylim(0,120)
remove_box()
tickParams()
gca().set_xticklabels([])

##############################################################################
### Stability ##########################################################
con='corr'
a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
c=data[2][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax3=fig.add_subplot(gsC[2])
vp=ax3.violinplot([list(a),list(b),list(c),list(d)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,c,d],widths=0.13,positions=[0,1,2,3],showcaps=False,showfliers=False,zorder=5,patch_artist=True)



colrs=['#56B4E9','#44AA99','#DC3C95','darkgray']
for i in range(4):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

remove_box()

gca().set_ylabel('Stability (r)')
gca().set_xticklabels([])
ylim(0,1.1)
gca().set_yticks([0,0.5,1])
gca().set_yticklabels([0,0.5,1])
tickParams()



##############################################################################
### Width ##########################################################
con='width'
a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
c=data[2][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax4=fig.add_subplot(gsC[3])

vp=ax4.violinplot([list(a),list(b),list(c),list(d)],showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,c,d],widths=0.13,positions=[0,1,2,3],showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','#DC3C95','darkgray']
for i in range(4):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)

remove_box()
ylim([deg2rad(45),deg2rad(228)])
gca().set_yticks([deg2rad(45),deg2rad(90),deg2rad(135),deg2rad(180),deg2rad(225)])
gca().set_yticklabels(['45\u00b0','90\u00b0','135\u00b0','180\u00b0','225\u00b0'])
gca().set_ylabel('Tuning width',labelpad=1)
gca().set_xticklabels([])
tickParams()

##############################################################################
### vlength ##########################################################
con='vlength'
a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
c=data[2][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax5=fig.add_subplot(gsC[4])

vp=ax5.violinplot([list(a),list(b),list(c),list(d)],widths=0.6,showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,c,d],widths=0.13,positions=[0,1,2,3],showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','#DC3C95','darkgray']
for i in range(4):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)


remove_box()
gca().set_ylabel('Vector length (r)')
gca().set_yticks([0,.5,1])
gca().set_yticklabels([0,.5,1])
ylim(0,1.02)
gca().set_xticklabels([r'WT$_L$','rd1','Gnat1/2',r'WT$_D$'])
tickParams()

##############################################################################
### Stability ##########################################################
con='info'
a=data[0][con].dropna().values.flatten()
b=data[1][con].dropna().values.flatten()
c=data[2][con].dropna().values.flatten()
d=data1[0][con].dropna().values.flatten()

ax6=fig.add_subplot(gsC[5])

vp=ax6.violinplot([list(a),list(b),list(c),list(d)],widths=0.6,showmeans=False, showmedians=False, showextrema=False,positions=[0,1,2,3])#widths=[0.14,0,0.82]
bp=boxplot([a,b,c,d],widths=0.13,positions=[0,1,2,3],showcaps=False,showfliers=False,zorder=5,patch_artist=True)

colrs=['#56B4E9','#44AA99','#DC3C95','darkgray']
for i in range(4):
   vp['bodies'][i].set_facecolor(colrs[i])
   vp['bodies'][i].set_edgecolor('white')
   vp['bodies'][i].set_alpha(0.7)
   bp['boxes'][i].set(facecolor='white')
   bp['medians'][i].set(color='k')
   bp['medians'][i].set(linewidth=1.5)
   
gca().set_xticklabels([r'WT$_L$','rd1','Gnat1/2',r'WT$_D$'])

gca().set_ylabel('Info. (bits/spk)')
remove_box()
tickParams()

# fig.savefig(r'C:/Users/kasum/Dropbox/ADn_Project/ADn_Figs/Figs/py_figs/Figure2S.pdf',dpi=600, format='pdf',transparent=True)
