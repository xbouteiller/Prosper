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

#%% shortened df
# dfjson=dfjson.iloc[3003:3015,:]
# df=df.iloc[:,:]

#%% merge and compute features
mdf2=MergeDFAndComputeFeature(df1=dfjson, df2=df)
print(mdf2)

# load df1 & df2 within class
mdf2.instantiate_df()

# remove www. before computing features
mdf2.clean_adress()

# merge df & compute feature
mdf2.mergedf()

#%%
print(mdf2.df_merged.shape)
print(mdf2.df_merged.head())


#%% preprocess for NLP
mdf2.nlp_preprocess(stop_fr=None, stop_en=None)

#%%
print(mdf2.df_merged.shape)
print(mdf2.df_merged.head())

#%%
print(mdf2.df_merged.columns)


#%%
print(mdf2.df_merged[mdf2.df_merged['language']=='fr'])

#%%

print(mdf2.df_merged[mdf2.df_merged['language']=='UNKNOWN'])

#%%
print(mdf2.df_merged.iloc[148,:])

#%%

print(mdf2.df_merged.language.value_counts())


#%%
mdf2.nlp_process(lang='en', min_df=2)


# %%
print(mdf2.tfidf_features)
print(mdf2.tfidf)
print(mdf2.tfidf.shape)

#%%
print(list(spacy.lang.fr.stop_words.STOP_WORDS)+[None])























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