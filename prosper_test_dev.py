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

#%%
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



#%%
count_notattributed=[]

for met in  ['braycurtis', 'canberra', 'chebyshev', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
    try:
        ml.do_clustering(eps=0.2, min_samples=5, metric=met)
        unique, counts = np.unique(ml.dfy_db, return_counts=True)
        count_notattributed.append(dict(zip(unique, counts))[-1])
    except:
        print(met)
        pass
    
print(count_notattributed)
#%%

























print(ml.X_train.bothnumsandwords)





#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(mdf2.tfidf,
                                                                 columns=mdf2.tfidf_features),
                                                    concat_df['wiki'],
                                                    random_state=99,
                                                    test_size=0.5)

#%%
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(ml.X_train, ml.y_train.values.ravel())
#%%
print(ml.y_train)
#%%
print(model.score(X_test, y_test))
#%%
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, model.predict(X_test)))


#%%
print(pd.DataFrame(mdf2.tfidf_features).reset_index())
print(pd.DataFrame(rf.feature_importances_).reset_index())
a=pd.DataFrame(mdf2.tfidf_features, columns = ['a']).reset_index()
b=pd.DataFrame(rf.feature_importances_,  columns = ['c']).reset_index()

#%%
print(pd.concat([a,b],axis=1).sort_values('c'))

a=pd.concat([a,b],axis=1).sort_values('c')[-100::]




#%%
print(dfjson[['snippet']])
#%%
print(type(dfjson[['snippet']]))

a=dfjson['snippet'].map(lambda x: nlp_floww(x))
#%%
print(pd.DataFrame(a))
#%%

def nlp_floww(text):
    import spacy
    from textblob import TextBlob
    import re

    b = TextBlob(text)
    lang = b.detect_language()
    
    if lang == 'en': 
        nlp=spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])
        stopwords=spacy.lang.en.stop_words.STOP_WORDS
    elif lang =='fr':
        nlp=spacy.load("fr_core_news_md", disable=["tagger", "parser", "ner"])
        stopwords=spacy.lang.fr.stop_words.STOP_WORDS
    else:
        print('identified language is neither english nor french')
        pass    
    
    text = text.lower()
    text=re.sub(r"///"," ",text)
    doc = nlp(text)

    # Generate lemmatized tokens
    lemmas = [token.lemma_ for token in doc]

    # Remove stopwords and non-alphabetic tokens
    a_lemmas = [lemma for lemma in lemmas
                if lemma.isalpha() and lemma not in stopwords]

    # Print string after text cleaning
    print(' '.join(a_lemmas),lang)
    return ' '.join(a_lemmas), lang
#%%
import re

text = 'aae///aezae///'
text=re.sub(r"///"," ",text)
print(text)

#%%
from textblob import TextBlob

# text = 'it is a///super try! Hoping, that will work...'
text = 'ceci est un autre essai où j\'ai mis dès caractères accentués hé hé hé & c\'est super'
b = TextBlob(text)
lang = b.detect_language()
print(lang)

#%%
a=nlp_flow(text)
print(a)