# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:05:30 2020

@author: sungw686
"""
##XGBOOST###################

import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import cartopy.crs as ccrs
from Operations import plotting, PickedForTest, plotPredictedTest, GridSearchX,GridSearch,Pairplotting
import time

from xgboost import plot_tree
import xgboost as xgb


Attempt='10th'
Run='39'
Res='05'
gridsearch='yes'

#Data_o = pd.read_csv('%sAttempt/Data/Combined_Gondwana_%s.txt' % (Attempt,Res), sep=' ')
Data_o = pd.read_csv('Data/Combined_GF_%s.txt' % (Res), sep=' ')
#Data_o = pd.read_csv('Data/Combined_%s.txt' % (Res), sep=' ')
Data_o.loc[(Data_o['Lon']==180) & (Data_o['Lat']==90), 'HF'] = np.nan
Data_o.loc[(Data_o['Lon']==-180) & (Data_o['Lat']==90), 'HF'] = np.nan
#Data_o.loc[(Data_o['Lat']>=-30) & (Data_o['Lat']<=30), 'Bz5']=-999
Data_o= Data_o.round({'Quality': 0, 'Tectonics': 0, 'Geology': 0})
    
    
plotting(Data_o.Lon,Data_o.Lat,Data_o.HF,'[mW/m$^2$]','Heat_Flow',0,120,Attempt,Run)
#plotting(Data_o.Lon,Data_o.Lat,Data_o.Bz5,'[nT]','Bz at 5km',-200,200)
#plotting(Data_o.Lon,Data_o.Lat,Data_o.iloc[:,3],'[Unit]','Geology',0,6)


Data_o = Data_o.dropna(subset=(['HF', 'Lon', 'Lat'])) 
M=np.vstack((Data_o.Lon, Data_o.Lat, Data_o.HF)).T
np.savetxt('%sAttempt/Binned_HF_measurements_%s.txt' % (Attempt,Run), M, fmt='%.3f')

Data_o.info()
Data_o.Topo = Data_o.Topo.fillna(-999)
Data_o.SI_TopoIso = Data_o.SI_TopoIso.fillna(-999)
Data_o.MeanCurv = Data_o.MeanCurv.fillna(-999)
Data_o.IceTopography = Data_o.IceTopography.fillna(-999)
Data_o.MeanCurvGrad = Data_o.MeanCurvGrad*1000
Data_o.LABGrad = Data_o.LABGrad*1000
Data_o.MohoGrad = Data_o.MohoGrad*1000
Data_o.TopoGrad = Data_o.TopoGrad*1000
Data_o.TectonicsGrad = Data_o.TectonicsGrad*1000
Data_o.SusGrad = Data_o.SusGrad*1000
Data_o.Bz5Grad = Data_o.Bz5Grad*1000



Data = Data_o[['HF','Lon','Lat', 'IsoCorrAnomaly', 'MeanCurv', 'MeanCurvGrad', 'SI_TopoIso', 
               'LAB', 'LAB_LitMod', 'LAB_LitMod_Aus17', 'LAB_AN1', 'LABGrad', 'Moho', 'Moho_AN1',
               'Moho_AN1_Aus17_Afr', 'Moho_LitMod_Aus17_Afr', 'MohoGrad', 'Topo',
               'TopoGrad', 'Tectonics', 'TectonicsGrad', 'Ridge', 'Transform', 'Trench',
               'YoungRift', 'Volcanos', 'MagAnomaly', 'Sus', 'Sus_AN1', 'Sus_AN1_Aus17_Afr',
               'Sus_LitMod_Aus17_Afr', 'SusGrad', 'Bz5', 'Bz5Grad', 'IceTopography', 'Curie']] 

#Data = Data_o[['HF','Lon','Lat', 'IsoCorrAnomaly', 'MeanCurv', 'SI_TopoIso', 
#               'LAB', 'Moho', 'Topo', 'Tectonics', 'Ridge', 'Transform', 'Trench',
#               'YoungRift', 'Volcanos', 'MagAnomaly', 'Sus',
#               'Bz5', 'IceTopography', 'Curie']] 
print(Data.head(0))
X=[]


Data = Data.loc[(Data['Topo']>=-1000.0) & (Data['Topo'] <=3000.0) | (Data['IceTopography']>0.0)]
#Data = Data.loc[(Data['Lon']>=-13) & (Data['Lon']<=40) & (Data['Lat']<=60) & (Data['Lat']>=35)] #Europe
#Data = Data.loc[(Data['Lon']>=110) & (Data['Lon']<=160) & (Data['Lat']<=-10) & (Data['Lat']>=-60)] #Australia
#Data = Data.loc[(Data['Lon']>=10) & (Data['Lon']<=160) & (Data['Lat']<=5) & (Data['Lat']>=-60)] #Australia & Africa
#Data = Data.loc[(Data['Lon']>=70) & (Data['Lon']<=160) & (Data['Lat']<=25) & (Data['Lat']>=-60)] #Australia & India
#Data = Data.loc[(Data['Lon']>=10) & (Data['Lon']<=160) & (Data['Lat']<=25) & (Data['Lat']>=-60)] #Australia & India & Africa

#ANTARCTICA
Data_Exept_ANT = Data.loc[(Data['Lat']>-60)]
Data_ANT = Data.loc[(Data['Lat']<=-60)]

##GREENLAND
#Data_Exept_ANT = Data#.loc[(Data['Lat']<58) & (Data['Lon']>-15) | (Data['Lat']>84) & (Data['Lon']<-70)]
#Data_ANT = Data.loc[(Data['Lat']<=84) & (Data['Lat']>=58) & (Data['Lon']<=-15) & (Data['Lon']>=-70)]

#plotting(Data_ANT.Lon,Data_ANT.Lat,Data_ANT.HF,'[mW/m$^2$]','Heat_Flow_not_Greenland',0,120,Attempt,Run)


plotting(Data.Lon,Data.Lat,Data.HF,'[mW/m$^2$]','Heat_Flow_filtered',0,120,Attempt,Run)

#
#Features = ['Tectonics', 'Bz5', 'Topo', 'MeanCurv', 'Sus', 'Transform', 'YoungRift',
#            'Trench', 'LAB', 'Ridge','Moho', 'Volcanos', 'Curie']


#Features = ['Topo','Tectonics','Sus_AN1_Aus17_Afr','Volcanos','Moho_AN1_Aus17_Afr',
#                           'Ridge', 'LAB_AN1', 'Trench','YoungRift',
#                           'Transform','MeanCurv','Bz5']

#Features = ['Volcanos','Moho_AN1_Aus17_Afr', 'Ridge', 'LAB_AN1', 'Trench',
#            'YoungRift', 'Transform','Sus_AN1_Aus17_Afr','MeanCurv', 'Topo', 'Bz5', 'Tectonics']

Features = ['Volcanos','Moho', 'Ridge', 'LAB', 'Trench','YoungRift']

#Features = ['Topo','Tectonics','Sus','Volcanos','Moho',
#                           'Ridge', 'LAB', 'Trench','YoungRift',
#                           'Transform','MeanCurv','Bz5']

#Features = ['Volcanos','Moho', 'Tectonics', 'YoungRift', 'Ridge', 'Trench', 'Sus',
#            'LAB', 'Transform', 'MeanCurv', 'Topo', 'Bz5']

#Features = ['Volcanos','Moho', 'MohoGrad', 'Tectonics', 'TectonicsGrad', 'YoungRift',
#            'Ridge', 'Trench', 'Sus', 'SusGrad', 'LAB', 'LABGrad', 'Transform',
#            'MeanCurv', 'MeanCurvGrad', 'Topo', 'TopoGrad', 'Bz5', 'Bz5Grad']



X=DataFrame(Data_Exept_ANT, columns=Features)
X_ANT=DataFrame(Data_ANT, columns=Features)

y=[]
y1=[]
for j in range(len(Data_Exept_ANT)):
    y.append(Data_Exept_ANT.iloc[j,0]) 
    
for j in range(len(Data_ANT)):
    y1.append(Data_ANT.iloc[j,0]) 
    
Mean = sum(y)/len(y)
print('Mean Heat Flow:', Mean)
print('Choice of HF measurements:', len(X))
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
x_train = x_train.append(X_ANT)
y_train =y_train + y1


t_0 = time.time()
if gridsearch == 'yes':    
    Best = GridSearchX(x_train,y_train)
    LR = Best['learning_rate']
    MD = Best['max_depth']
    SS = Best['min_samples_split']
    Su = Best['subsample']
    #GA = Best['gamma']
else:
    LR = 0.01
    MD = 11
    SS = 5
    Su = 0.5
t_1 = time.time()
print ("It took %.1f seconds" % (t_1-t_0))

Gamma=120
print('Learning Rate:', LR, 'Maximum Depth', MD, 'Subsample', Su, 'min samples split', SS)
params = {'objective':'reg:squarederror','learning_rate':LR,'max_depth':MD,
          'n_estimators':1000,'subsample':Su, 'gamma': Gamma, 'min_samples_split': SS}
model = xgb.XGBRegressor(**params)
model.fit(x_train,y_train)

#SCORES########################################################################
model_score=model.score(x_train,y_train)
y_pred=model.predict(x_test)
ms_error=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
RMSE=np.sqrt(ms_error)#/Mean
M=np.vstack((model_score, ms_error, r2, RMSE, LR, MD, SS)).T
np.savetxt('%sAttempt/Scores_%s.txt' % (Attempt,Run), M, fmt='%.3f', 
           header='modelScore MSerror r2 RMSE LearningRate MaxDepth minSampleSplit')

## define the evaluation method
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
## report performance
#print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

Corr=np.vstack((y_test, y_pred)).T
np.savetxt('%sAttempt/Y_%s.txt' % (Attempt,Run), Corr, fmt='%.3f', 
           header='Actual Predicted')


print('model score:', model_score)
print('Mean squared error:', ms_error)
print('Test Variance score (r2):', r2)
print('RMSE:', RMSE)
###############################################################################
plotPredictedTest(y_test,y_pred,Attempt,Run)
#Topo = PickedForTest(11,x_test)
#plotPredictedTest(y_test,y_pred,Attempt,Run,Topo,min(Topo)/1.4,max(Topo)/1.4,'Topo')

#PLOT IMPORTANCES##############################################################
names= ['Topo', 'Tectonics','Sus','Volcanos', 'Moho','Ridge', 'LAB', 'Trench','YoungRift',
        'Transform','MeanCurv','Bz5']
#names= ['Volcanos','Moho', 'Tectonics', 'Young Rift', 'Ridges', 'Trench', 'Susceptibility',
#        'LAB', 'Transform Fault', 'Mean Curvature', 'Topography', 'Bz']
#names= ['Volcanos','Moho', 'Ridge', 'LAB', 'Trench',
#        'YoungRift', 'Transform','Susceptibility','Mean Curvature', 'Topography', 'Bz5', 'Tectonics']
#names= ['Volcanos','Moho', 'Moho Gradient', 'Tectonics', 'Tectonic Gradient',
#        'Young Rift', 'Ridges', 'Trench', 'Susceptibility', 'Susceptibility Gradient',
#        'LAB', 'LAB Gradient', 'Transform Fault', 'Mean Curvature', 'Mean Curvature Gradient',
#        'Topography', 'Topography Gradient', 'Bz', 'Bz Gradient']


name = []
feature_importance = model.feature_importances_
#feature_importance = 100*(feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

#fig = plt.figure(figsize=(9, 6))
fig = plt.figure(figsize=(12, 8))
plt.barh(pos, feature_importance[sorted_idx], align='center')
for i in range(len(sorted_idx)):
    name.append(names[sorted_idx[i]])
plt.yticks(pos, name[:])
plt.xlabel('Importance',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.title('Variable Importance',fontsize=20)
plt.tight_layout()
fig.savefig('%sAttempt/Importance_Deviance_%s.jpg' % (Attempt,Run), dpi=300)
plt.show()


#plt.rcParams['figure.figsize'] = [80, 50]
#xgb.plot_tree(model,num_trees=0,rankdir='LR').get_figure().savefig('%sAttempt/Tree1st_%s.jpg' % (Attempt,Run), dpi=300)
#plt.close()
#xgb.plot_tree(model,num_trees=999,rankdir='LR').get_figure().savefig('%sAttempt/TreeLast_%s.jpg' % (Attempt,Run), dpi=300)
#plt.close()


##PREDICT HEATFLUX##############################################################
ToPredict = pd.read_csv('Data/Grid_%s.txt' % (Res), sep=' ')
ToPredict.Topo = ToPredict.Topo.fillna(-999)
ToPredict.MeanCurv = ToPredict.MeanCurv.fillna(-999)
ToPredict.SI_TopoIso = ToPredict.SI_TopoIso.fillna(-999)
ToPredict.IceTopography = ToPredict.IceTopography.fillna(-999)
ToPredict=ToPredict.round({'Tectonics': 0})
ToPredict.MeanCurvGrad = ToPredict.MeanCurvGrad*1000
ToPredict.LABGrad = ToPredict.LABGrad*1000
ToPredict.MohoGrad = ToPredict.MohoGrad*1000
ToPredict.TopoGrad = ToPredict.TopoGrad*1000
ToPredict.TectonicsGrad = ToPredict.TectonicsGrad*1000
ToPredict.SusGrad = ToPredict.SusGrad*1000
ToPredict.Bz5Grad = ToPredict.Bz5Grad*1000



#TP = ToPredict[['Lon','Lat', 'IsoCorrAnomaly', 'MeanCurv', 'SI_TopoIso', 
#               'LAB', 'LAB_LitMod', 'LAB_LitMod_Aus17', 'Moho', 'Moho_AN1',
#               'Moho_AN1_Aus17_Afr', 'Topo', 'Tectonics', 'Ridge', 'Transform', 'Trench',
#               'YoungRift', 'Volcanos', 'MagAnomaly', 'Sus', 'Sus_AN1', 'Sus_AN1_Aus17_Afr',
#               'Bz5', 'IceTopography', 'Curie']]


TP = ToPredict[['Lon','Lat', 'IsoCorrAnomaly', 'MeanCurv', 'MeanCurvGrad', 'SI_TopoIso', 
               'LAB', 'LAB_LitMod', 'LAB_LitMod_Aus17', 'LAB_AN1', 'LABGrad', 'Moho', 'Moho_AN1',
               'Moho_AN1_Aus17_Afr', 'Moho_LitMod_Aus17_Afr', 'MohoGrad', 'Topo',
               'TopoGrad', 'Tectonics', 'TectonicsGrad', 'Ridge', 'Transform', 'Trench',
               'YoungRift', 'Volcanos', 'MagAnomaly', 'Sus', 'Sus_AN1', 'Sus_AN1_Aus17_Afr',
               'Sus_LitMod_Aus17_Afr', 'SusGrad', 'Bz5', 'Bz5Grad', 'IceTopography', 'Curie']] 

#TP = ToPredict[['Lon','Lat', 'IsoCorrAnomaly', 'MeanCurv', 'SI_TopoIso', 
#               'LAB', 'Moho', 'Topo', 'Tectonics', 'Ridge', 'Transform', 'Trench',
#               'YoungRift', 'Volcanos', 'MagAnomaly', 'Sus',
#               'Bz5', 'IceTopography', 'Curie']]

print(TP.head(0))

TP = TP.loc[(TP['Topo']>=-1000.0) & (TP['Topo'] <=3000.0) | (TP['IceTopography']>0.0)]

X_new=DataFrame(TP, columns=Features)


Lon_p=[]
Lat_p=[]
for i in range(len(TP)):
    Lon_p.append(TP.iloc[i,0])
    Lat_p.append(TP.iloc[i,1])
new_pred = model.predict(X_new)


#    DF = pd.DataFrame({'Predicted HF \n [mW/m$^2$]':new_pred, 'MeanCurv':X_new.MeanCurv,
#                       'LAB \n [km]': X_new.LAB, 'Moho \n [km]': X_new.Moho, 'Topo \n [m]': X_new.Topo,
#                       'Tectonics \n [km]': X_new.Tectonics, 'Ridge \n [m]': X_new.Ridge,
#                       'Transform \n [m]': X_new.Transform, 'Trench \n [m]': X_new.Trench,
#                       'Young Rift \n [m]': X_new.YoungRift,'Volcanos \n [m]': X_new.Volcanos,
#                       'Sus \n [m]': X_new.Sus,'Bz \n [m]': X_new.Bz5})
#    DF=DF[DF.MeanCurv!=-999]    
#    Pairplotting(DF,Attempt,Run)
plotting(Lon_p,Lat_p,new_pred,'[mW/m$^2$]','Predicted_Heat_Flux',0,120,Attempt,Run)
M=np.vstack((Lon_p,Lat_p, new_pred)).T
np.savetxt('%sAttempt/Predicted_HF_%s.txt' % (Attempt,Run), M, fmt='%.3f')


