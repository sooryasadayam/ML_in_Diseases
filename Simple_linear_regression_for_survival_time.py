# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 05:41:37 2021

@author: Soorya
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import copy
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data_tumor = pd.read_csv("C:/Users/Soorya/WFH/Data/BreastCancer/tumor_BreastCancer.tsv", sep = '\t', header = 2)
data_normal = pd.read_csv("C:/Users/Soorya/WFH/Data/BreastCancer/normal_BreastCancer.tsv", sep = '\t', header = 2)
clinical = pd.read_csv("C:/Users/Soorya/WFH/Data/BreastCancer/clinical.tsv",sep='\t')

tcga_only=pd.DataFrame([clinical.iloc[i,:] for i,ID in zip(range(clinical.shape[0]),clinical.case_submitter_id) if ID[0:4]=='TCGA']) #Choose only from project TCGA 
tcga_only.drop_duplicates('case_submitter_id',inplace=True)
tcga_only=tcga_only.loc[:,['case_submitter_id','days_to_last_follow_up','days_to_death','vital_status']]
tcga_only.index=range(tcga_only.shape[0])
tcga_only["survival_time"]=[int(time)/365.0 if time!="'--" else int(tcga_only.loc[i,'days_to_death'])/365.0 for time,i in zip(tcga_only.days_to_last_follow_up,range(tcga_only.shape[0]))]
tcga_only["event"]= (tcga_only.vital_status=='Dead').astype(int)
tcga_only.set_index('case_submitter_id',drop=True,inplace=True)

data_tumor.drop('gene_name',axis=1,inplace=True)
data_normal.drop('gene_name',axis=1,inplace=True)

features_tumor = copy.deepcopy(data_tumor)
features_normal = copy.deepcopy(data_normal)

features_tumor.insert(0,'Condition',['Tumor']*features_tumor.shape[0])
features_normal.insert(0,'Condition',['Normal']*features_normal.shape[0])

features_tumor.set_index(features_tumor.columns[-1],drop=True,inplace=True)
features_normal.set_index(features_normal.columns[-1],drop=True,inplace=True)

x=pd.merge(tcga_only,features_tumor,left_index=True,right_index=True)
X=x.loc[~x.index.duplicated(keep='first')] #drop duplicate indices
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,6:],X.loc[:,'survival_time'],
                                                    stratify=X.loc[:,'event'],
                                                    test_size=0.33,random_state=42) # Maintain proportion of dead patients while splitting

#Check if stratification works
#print(np.mean(X.loc[X_train.index,'event']))
#print(np.mean(X.loc[X_test.index,'event']))

scaler_x=StandardScaler()
scaler_y=StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.to_frame())
X_test_scaled  = scaler_x.transform(X_test)                         # is it fit_transform or transform
y_test_scaled  = scaler_y.transform(y_test.to_frame())

# Linear_model = linear_model.LinearRegression()
# Linear_model.fit(X_train_scaled,y_train_scaled)
# rmse_train = mean_squared_error(y_train_scaled,Linear_model.predict(X_train))
# rmse_test  = mean_squared_error(y_test_scaled,Linear_model.predict(X_test))
Alphas = np.logspace(-5,0,num=10)
RMSE_train = []
RMSE_test  = []
for a in Alphas:
    Linear_model = linear_model.ElasticNet(alpha=a,l1_ratio=0.5, max_iter=10000)
    Linear_model.fit(X_train_scaled,y_train_scaled)
    rmse_train = mean_squared_error(y_train_scaled,Linear_model.predict(X_train_scaled))**0.5
    RMSE_train.append(rmse_train)
    rmse_test  = mean_squared_error(y_test_scaled,Linear_model.predict(X_test_scaled))**0.5
    RMSE_test.append(rmse_test)
    
plt.plot(Alphas, RMSE_train, color='r', label='Training_RMSE')
plt.plot(Alphas, RMSE_test, color='g', label='Test_RMSE')
plt.xlabel(r"$\alpha$")
plt.ylabel("RMSE")
plt.title("Finding optimal regularisation weight")
plt.legend()
plt.show()