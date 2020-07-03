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

# remove www. befrore computing features
mdf2.clean_adress()

# merge df & compute feature
mdf2.mergedf()

print(mdf2.df1.shape)
print(mdf2.df_merged.shape)
print(mdf2.df_merged.snippet)


#%%

def nlp_flow(text):
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
    print(' '.join(a_lemmas))
    return ' '.join(a_lemmas)
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
nlp_flow(text)