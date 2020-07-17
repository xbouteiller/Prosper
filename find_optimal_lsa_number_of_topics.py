import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import spacy

'''
python script to compute LSA and printing topic related words

Don't know why, but seems to work only in python script mode
>>> python find_optimal_lsa_number_of_topics.py
'''

if __name__ == '__main__':
	os.chdir('C:\\Users\\xavier\\Documents\\Prosper')
	from featuring.functions import describedf, StringAnalyzer, WebSiteListAnalyser, MergeDFAndComputeFeature

	#%load df
	dfjson=pd.read_json("data\\bing_results.json")
	df=pd.read_csv("data\\prop_wiki.csv")


	# REDUCE DIM OF DF FOR TESTING
	#------------------------------------------------
	# dfjson=dfjson.iloc[0:100,:]
	# df=df.iloc[0:100,:]
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

	#%% create a specific model
	mdf2.create_gensim_lsa_model(number_of_topics=10, words=10,lang='fr')

	#%% find optimal number of topics
	mdf2.plot_graph(start=2,stop=30,step=1)

	#
	# mdf2.create_gensim_lsa_model(number_of_topics=3, words=20, lang='fr')
