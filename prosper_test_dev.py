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
mdf2.nlp_process(lang='fr', min_df=100, max_df=2000, ngram_range=(1,1))

#%% inspect results
# print(mdf2.tfidf_features)
print(mdf2.tfidf)
print(mdf2.tfidf.shape)
word100=pd.DataFrame(mdf2.tfidf,columns=mdf2.tfidf_features).sum().sort_values()[-100::]
print(word100)               
      
#%% prepare data set
mdf2.preparedataset()

#%% split train evaluate classifier
from featuring.machinelearning import MachineLearning

ml = MachineLearning(dfX=mdf2.dfX, dfy=mdf2.dfy)

ml.split_data(random_state=99, 
              test_size=0.5, 
              stratify=None)

ml.instantiate_classif(classifier='rf',
                       max_depth=15,
                       class_weight=None,
                       n_estimators=500,
                       penalty=None,
                       C=None,
                       solver=None
                       )
ml.fit_classif()



#%% clustering
count_notattributed=[]

for met in  ['braycurtis', 'canberra', 'chebyshev', 'dice', 'jaccard', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean']:
   for eps in [0.01,0.05,0.1, 0.25,0.5,1]:
        try:
            ml.do_clustering(eps=eps, min_samples=5, metric=met)
            unique, counts = np.unique(ml.dfy_db, return_counts=True)
            count_notattributed.append({met:[eps, dict(zip(unique, counts))[-1],dict(zip(unique, counts))[0]]})
        except:
            print(met)
            pass
    
print(count_notattributed)
#%%
ml.do_clustering(eps=0.5, min_samples=5, metric='chebyshev')
unique, counts = np.unique(ml.dfy_db, return_counts=True)
print(dict(zip(unique, counts)))

#%%












