# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:21:16 2020

@author: xavier
"""

#%% import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import spacy

#%% import package perso

os.chdir('C:\\Users\\xavier\\Documents\\Prosper')
from featuring.functions import describedf, StringAnalyzer, WebSiteListAnalyser, MergeDFAndComputeFeature

#%% load df
dfjson=pd.read_json("data\\bing_results.json")
df=pd.read_csv("data\\prop_wiki.csv")

#%% merge and compute features
mdf2=MergeDFAndComputeFeature(df1=dfjson, df2=df)
print(mdf2)

# load df1 & df2 within class
mdf2.instantiate_df()

# remove www. before computing features
mdf2.clean_adress()

# merge df & compute feature
mdf2.mergedf()

#%% inspect results
print(mdf2.df_merged.shape)
print(mdf2.df_merged.head())


#%% preprocess for NLP
mdf2.nlp_preprocess(stop_fr=None, stop_en=None)

#%% inspect results
print(mdf2.df_merged.shape)
print(mdf2.df_merged.head())
print(mdf2.df_merged.columns)


#%% inspect results
print(mdf2.df_merged[mdf2.df_merged['language']=='fr'])
print(mdf2.df_merged[mdf2.df_merged['language']=='fr'].shape)

#%% inspect results
print(mdf2.df_merged[mdf2.df_merged['language']=='UNKNOWN'])
print(mdf2.df_merged.iloc[148,:])

#%% inspect results
print(mdf2.df_merged.language.value_counts())

#%% process for NLP
mdf2.nlp_process(lang='fr', min_df=50, max_df=500, ngram_range=(1,2))

#%% inspect results
# print(mdf2.tfidf_features)
print(mdf2.tfidf)
print(mdf2.tfidf.shape)
word100=pd.DataFrame(mdf2.tfidf,columns=mdf2.tfidf_features).sum().sort_values()[-100::]
print(word100)               
      
#%% prepare data set
mdf2.preparedataset(add_user_feature=False)

#%% split train evaluate classifier
from featuring.machinelearning import MachineLearning
#%% ML
ml = MachineLearning(dfX=mdf2.dfX, dfy=mdf2.dfy)

ml.split_data(random_state=99, 
              test_size=0.5, 
              stratify=None)

ml.instantiate_classif(classifier='lr',
                       max_depth=15,
                       class_weight=None,
                       n_estimators=500,
                       penalty=None,
                       C=3,
                       solver=None
                       )
ml.fit_classif()

#%%


#%% clustering
ml.find_dbscan(metrics=['euclidean', 'canberra'], eps=[0.2,0.5], min_samples=5)

#%%
if True:
    count_notattributed=[]
    
    for met in  ['braycurtis', 'canberra', 'chebyshev', 'dice','euclidean', 'jaccard', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean']:
       for eps in [0.1,0.25,0.5,1]:
            try:
                ml.do_dbscan(eps=eps, min_samples=5, metric=met)
                unique, counts = np.unique(ml.dfy_db, return_counts=True)
                count_notattributed.append({met:[eps, dict(zip(unique, counts))[-1],dict(zip(unique, counts))[0]]})
            except:
                print(met)
                pass
        
    print(count_notattributed)
    
#%%
score_dbscan=pd.DataFrame({'metric':[],
                           'eps':[],
                           'nerror':[],
                           'nc1':[]})
   
for c in count_notattributed:
    # print(list(c.items())[0][0], list(c.items())[0][1][0], list(c.items())[0][1][1],  list(c.items())[0][1][2] )
    score_dbscan=score_dbscan.append(pd.DataFrame({'metric':[list(c.items())[0][0]],
                                                   'eps':[list(c.items())[0][1][0]],
                                                   'nerror': [list(c.items())[0][1][1]],
                                                   'nc1': [list(c.items())[0][1][2]]}))
    
score_dbscan['sum_error_C1']=score_dbscan[['nerror', 'nc1']].sum(axis=1)
score_dbscan['Percentage_Err_Grp1']=score_dbscan['sum_error_C1']/ml.dfX.shape[0]
print(score_dbscan.sort_values(['Percentage_Err_Grp1','nerror', 'nc1'], ascending=[True,False, True]).drop(['nerror', 'nc1', 'sum_error_C1'],axis=1))
print('\n--------------------------------------------------') 
print('\nPercentage represents the proportion of rows assignated either to Error group or to only One cluster by dbscan \nHigh ratio indicates non consistent clustering')



score_dbscan.set_index(score_dbscan["metric"]+score_dbscan["eps"].astype(str))[['Percentage_Err_Grp1']].plot.bar(rot=90)

#%%
ml.do_dbscan(eps=0.5, min_samples=5, metric='chebyshev')
unique, counts = np.unique(ml.dfy_db, return_counts=True)
print(dict(zip(unique, counts)))
cluster_dbscan=pd.DataFrame(ml.dfy_db)
#%%
print(pd.crosstab(ml.dfy_db, mdf2.df_merged.loc[ mdf2.df_merged['language']=='fr', ['wiki']].values.ravel()))
#%%

if False:
    ml.find_kmeans(max_k = 10)

#%%
ml.do_kmeans(nK=2)
cluster_kmeans=pd.DataFrame(ml.dfy_kmeans)
#%%
print('\n------row: kmeans, col: wiki------')
print(pd.crosstab(ml.dfy_kmeans, mdf2.df_merged.loc[ mdf2.df_merged['language']=='fr', ['wiki']].values.ravel()))

#%%
print('\n------row: dbscan, col: kmeans------')
print(pd.crosstab(ml.dfy_db, ml.dfy_kmeans))

#%% ML with kmeans clusters 


ml = MachineLearning(dfX=mdf2.dfX, dfy=cluster_kmeans)

ml.split_data(random_state=99, 
              test_size=0.5, 
              stratify=None)

ml.instantiate_classif(classifier='lr',
                       max_depth=4,
                       class_weight=None,
                       n_estimators=500,
                       penalty=None,
                       C=None,
                       solver=None
                       )
ml.fit_classif()

print('\npca done? ', ml.pca)

#%%

coef_lr=pd.DataFrame.from_dict({'name':mdf2.dfX.columns,'coef':(ml.classif.coef_[0].ravel())})

print(coef_lr.sort_values('coef', ascending = False)[0:30])
print(coef_lr.sort_values('coef', ascending = False)[-30::])
#%%
coef_lr.sort_values('coef', ascending = False)['coef'].plot.bar()
plt.show()
#%% ML with PCA


ml = MachineLearning(dfX=mdf2.dfX, dfy=mdf2.dfy)

ml.split_data(random_state=99, 
              test_size=0.5, 
              stratify=None)

ml.do_pca()

ml.instantiate_classif(classifier='lr',
                       max_depth=15,
                       class_weight=None,
                       n_estimators=500,
                       penalty='l2',
                       C=17,
                       solver=None
                       )
ml.fit_classif()

print('\npca done? ', ml.pca)




#%% clustering with PCA

ml = MachineLearning(dfX=mdf2.dfX, dfy=mdf2.dfy)
ml.do_pca()


if True:
    count_notattributed=[]
    
    for met in  ['chebyshev', 'jaccard','euclidean']:
       for eps in [0.5,1,2]:
            try:
                ml.do_dbscan(eps=eps, min_samples=5, metric=met)
                unique, counts = np.unique(ml.dfy_db, return_counts=True)
                count_notattributed.append({met:[eps, dict(zip(unique, counts))[-1],dict(zip(unique, counts))[0]]})
            except:
                print(met)
                pass
        
    print(count_notattributed)


if True:
    ml.find_kmeans(max_k = 10)


ml.do_kmeans(nK=2)

ml.do_dbscan(eps=0.5, min_samples=5, metric='chebyshev')
unique, counts = np.unique(ml.dfy_db, return_counts=True)
print(dict(zip(unique, counts)))

print('\n------row: dbscan, col: wiki------')
print(pd.crosstab(ml.dfy_db, mdf2.df_merged.loc[ mdf2.df_merged['language']=='fr', ['wiki']].values.ravel()))
print('\n')

print('\n------row: kmeans, col: wiki------')
print(pd.crosstab(ml.dfy_kmeans, mdf2.df_merged.loc[ mdf2.df_merged['language']=='fr', ['wiki']].values.ravel()))
print('\n')

print('\n------row: dbscan, col: kmeans------')
print(pd.crosstab(ml.dfy_db, ml.dfy_kmeans))
print('\n')

