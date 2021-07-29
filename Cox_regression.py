# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 07:06:59 2021

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
from lifelines.datasets import load_rossi
from lifelines import KaplanMeierFitter
import copy

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
pca = PCA(n_components=4)
scaler = MinMaxScaler()
projections_pca= pca.fit_transform(scaler.fit_transform(features_tumor.iloc[:,1:]))
projections_pca= pd.DataFrame(projections_pca,index=features_tumor.index,columns=['PC1','PC2','PC3','PC4'])
final_data_set = pd.merge(tcga_only.loc[:,['survival_time','event']],
                          projections_pca,left_index=True,right_index=True)
final_data_set = final_data_set.loc[~final_data_set.index.duplicated(keep='first'),:]

kmf = KaplanMeierFitter()
kmf.fit(tcga_only.survival_time, event_observed=tcga_only.event)
kmf.plot_survival_function()
plt.xlabel('Time (years)')
plt.ylabel('Probability of Survival')

plt.figure()
cph = CoxPHFitter()
cph.fit(final_data_set, duration_col='survival_time', event_col='event')
cph.print_summary()  # access the individual results using cph.summary
cph.plot()

# plt.figure()
# coefficients=pca.components_[2]
# x=np.c_[data_normal.columns[0:-1],coefficients]
# x=pd.DataFrame(x[:,1],index=x[:,0])
# x.sort_values(0,axis=0)
# plt.hist(x,100) # histogram of the 60,000+ coefficients of PC-2