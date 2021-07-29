# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 18:09:26 2021

@author: Soorya
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# from google.colab import drive
# drive.mount('/content/drive')

data_tumor = pd.read_csv("C:/Users/Soorya/WFH/Data/BreastCancer/tumor_BreastCancer.tsv", sep = '\t', header = 2)
data_normal = pd.read_csv("C:/Users/Soorya/WFH/Data/BreastCancer/normal_BreastCancer.tsv", sep = '\t', header = 2)

#data_tumor.head()
data_tumor.drop('gene_name',axis=1,inplace=True)
data_normal.drop('gene_name',axis=1,inplace=True)


features_tumor = data_tumor
features_normal = data_normal
features_tumor.insert(0,'Condition',['Tumor']*features_tumor.shape[0])
features_normal.insert(0,'Condition',['Normal']*features_normal.shape[0])

scaler = MinMaxScaler()#StandardScaler() #
# scaled_features_tumor = scaler.fit_transform(data_tumor.iloc[:,1:-1])
# scaled_features_normal = scaler.fit_transform(data_normal.iloc[:,1:-1])

# full_matrix_tumor=pd.DataFrame(np.c_[features_tumor.iloc[:,0],scaled_features_tumor,features_tumor.iloc[:,-1]], #Last column contains abbreviated patient IDs
#                                        index=features_tumor.index,columns=features_tumor.columns) 
# full_matrix_normal=pd.DataFrame(np.c_[features_normal.iloc[:,0],scaled_features_normal,features_normal.iloc[:,-1]],
#                                        index=features_normal.index,columns=features_normal.columns) 

features_all = pd.DataFrame(np.c_[np.r_[features_normal.iloc[:,0],features_tumor.iloc[:,0]],
                                  scaler.fit_transform(pd.concat([features_normal.iloc[:,1:-1],features_tumor.iloc[:,1:-1]],axis=0)),
                                  ],index=np.r_[features_normal.iloc[:,-1],features_tumor.iloc[:,-1]],columns=features_normal.columns[0:-1])
# features_all=pd.concat([full_matrix_normal,full_matrix_tumor],axis=0,ignore_index=True) #features are min-max scaled
# features_all.set_index(features_all.columns[-1],drop=True,inplace=True) #set the last column as index and drop the last column

tsne = TSNE(n_components=2, random_state=42)
projections = tsne.fit_transform(features_all.iloc[:,1:])
pre_plot=pd.DataFrame(np.c_[features_all.Condition,projections],
             index=features_all.index,columns=['Condition','x','y'])
plt.figure()
sns.relplot(x='x',y='y',hue='Condition',data=pre_plot)
plt.xlabel('t-SNE-1')
plt.ylabel('t-SNE-2')

import umap
reducer=umap.UMAP(random_state=42)
embedding = reducer.fit_transform(features_all.iloc[:,1:])
pre_plot_umap=pd.DataFrame(np.c_[features_all.Condition,embedding],
             index=features_all.index,columns=['Condition','x','y'])
plt.figure()
sns.relplot(x='x',y='y',hue='Condition',data=pre_plot_umap)
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')

pca = PCA(n_components=2)
projections_pca= pca.fit_transform(features_all.iloc[:,1:])
pre_plot_pca=pd.DataFrame(np.c_[features_all.Condition,projections_pca],
             index=features_all.index,columns=['Condition','x','y'])

plt.figure()
sns.relplot(x='x',y='y',hue='Condition',data=pre_plot_pca)
plt.xlabel('PC-1')
plt.ylabel('PC-2')