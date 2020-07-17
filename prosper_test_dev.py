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


dfjson=dfjson.iloc[0:100,:]
df=df.iloc[0:100,:]
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


#%% clustering
ml = MachineLearning(dfX=mdf2.dfX, dfy=mdf2.dfy)

if True:
  ml.find_dbscan(metrics=['jaccard','sqeuclidean', 'chebyshev'], eps=[0.01,0.1,1,10], min_samples=4)

if False:
    ml.find_kmeans(max_k = 20)


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

cluster_kmeans=pd.DataFrame(ml.dfy_kmeans)
cluster_dbscan=pd.DataFrame(ml.dfy_db)

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

coef_lr.sort_values('coef', ascending = False)['coef'].plot.bar()
plt.show()


#%%

plt.scatter(mdf2.dfX['assurance'], mdf2.dfX['puissance'])
plt.show()

#%%
plt.scatter(mdf2.dfX['marque'], mdf2.dfX['modèle'])
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
                       penalty=None,
                       C=17,
                       solver=None
                       )
ml.fit_classif()

print('\npca done? ', ml.pca)



#%% clustering with PCA

ml = MachineLearning(dfX=mdf2.dfX, dfy=mdf2.dfy)
ml.do_pca()


if True:
  ml.find_dbscan(metrics=['braycurtis','sqeuclidean', 'chebyshev'], eps=[0.01,0.2,0.7], min_samples=4)

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

#%%  lsa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import spacy

if __name__ == '__main__':

    os.chdir('C:\\Users\\xavier\\Documents\\Prosper')
    from featuring.functions import describedf, StringAnalyzer, WebSiteListAnalyser, MergeDFAndComputeFeature
    
    
    #%load df
    dfjson=pd.read_json("data\\bing_results.json")
    df=pd.read_csv("data\\prop_wiki.csv")
    
    # REDUCE DIM OF DF FOR TESTING
    #------------------------------------------------
    # dfjson=dfjson.iloc[0:1000,:]
    # df=df.iloc[0:1000,:]
    #------------------------------------------------
    
    
    #%merge and compute features
    mdf2=MergeDFAndComputeFeature(df1=dfjson, df2=df)
    print(mdf2)
    
    # load df1 & df2 within class
    mdf2.instantiate_df()
    
    # remove www. before computing features
    mdf2.clean_adress()
    
    # merge df & compute feature
    mdf2.mergedf()
    
    # preprocess for NLP
    mdf2.nlp_preprocess(stop_fr=None, stop_en=None)
    
    #%%
    mdf2.create_gensim_lsa_model(number_of_topics=3, words=20,lang='fr')
 
    #%%
    mdf2.extract_words_by_lsa_topic( num_words=3)
#%%
    mdf2.create_gensim_lda_model(number_of_topics=5, words=20,lang='fr')
    #%%
    mdf2.extract_words_by_lda_topic(num_words=10)
    
    #%%
    mdf2.lda_predicted_cluster()
    print(mdf2.lda_cluster)