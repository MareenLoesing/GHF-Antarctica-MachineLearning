# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:39:39 2020

@author: Mareen Loesing
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

def plotting(Lon,Lat,value,unit,name,a,b,Plot_file,Run):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    plt.scatter(Lon,Lat,c=value,cmap='jet', marker='s', s=5, vmin=a, vmax=b)
    cb = plt.colorbar(fraction=0.021, pad=0.04)
    cb.set_label('%s' % unit, labelpad=30, fontsize=15, rotation=270) 
    plt.title('%s' % name, fontsize=15)
    cb.ax.tick_params(labelsize=15)
    plt.show()
    fig.savefig('%s/%s_%s.jpg' % (Plot_file,name,Run), dpi=300)
       
def GridSearch(x,y):
    param_grid={'learning_rate':[0.01,0.02],
           'max_depth':[8,10,12],
           'min_samples_split':[20,30,40,50]}
    est = GradientBoostingRegressor(n_estimators=1000)
    gs_cv = GridSearchCV(est, param_grid, cv=5).fit(x,y)
    return gs_cv.best_params_
     
def GridSearchX(x,y):
#    params={'learning_rate':[0.01,0.015],'max_depth':[6,10,11],
#        'subsample':[0.5,0.7,0.8,1],'min_samples_split':[5,10,20]}
    params={'learning_rate':[0.01,0.001],'max_depth':[6,10,11],
        'subsample':[0.5,0.7,0.8,1],'min_samples_split':[5,10,20]}
    model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=1000)
    grid=GridSearchCV(model,params, cv=5,n_jobs=-1).fit(x,y)
    return grid.best_params_

    
    
def PickedForTest(number,x_test):
    Feature = []
    for i in range(len(x_test)):
        Feature.append(x_test.iloc[i,number])
    return Feature
    

def plotPredictedTest(x,y,Plot_file,Run,value=None,d=None,e=None):
    fig, ax = plt.subplots(figsize=(9,7))
    if not value is None:
        cmap = plt.get_cmap('gist_earth')
        plt.scatter(x,y,c=value,cmap=cmap,edgecolors=(0,0,0),vmin=d,vmax=e)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        cb.set_label('[m]',fontsize=15, rotation=270)
    else:
        plt.scatter(x,y, edgecolors=(0,0,0))
    plt.plot([min(x),max(x)], [min(x),max(x)], 'k--', lw=2)
    plt.xlabel('Actual'+ '\n' + '[mW/m$^2$]',fontsize=20)
    plt.ylabel('Predicted'+ '\n' + '[mW/m$^2$]',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title('Ground Truth vs Predicted',fontsize=20)
    plt.tight_layout()
    fig.savefig('%s/Test-Pedicted_%s.jpg' % (Plot_file,Run), dpi=300)
    plt.show()
    
    
    
    
    
    
def Pairplotting(DF,Plot_file,Run):
    plt.rcParams['figure.figsize']=40,40
    sns.set_context("paper", rc={"font.size":19,"axes.labelsize":12}) 
    sns.set("notebook", rc={'figure.figsize':(40,40)}, font_scale=1.6)
    sns_plot = sns.PairGrid(DF) 
    sns_plot.fig.subplots_adjust(hspace=0.1, wspace=0.1)
#    sns_plot.axes[1,0].set_ylim((-12,2))
#    sns_plot.axes[12,1].set_xlim((-12,2))
    sns_plot.map_lower(sns.kdeplot, cmap="jet", shade=True, n_levels=100, shade_lowest=False)
    sns_plot.map_diag(sns.distplot, kde=False)
    sns_plot.map_upper(plt.scatter, edgecolor="white", color="#626262")
    sns_plot.savefig("%s/Pairplot_%s" % (Plot_file,Run))
    
    
    
    
    
