def describedf(df,nco=1):
    '''
    Small function useful for describing a dataframe
    '''
    print('shape is nrow:{row} and ncol:{col}'.format(row=df.shape[0],col=df.shape[1]), '\n')
    print(df.head(), '\n')
    [print('type of col:{} is:{}'.format(df.columns[nc],df.iloc[:,nc].dtypes), '\n') for nc in range(0,nco)]
    

class StringAnalyzer():
    
    '''
    class for computing features on a string
    
    a word in the provided string is a string entity separated from other entity by one of this separator ".:/-"
    
    methods:
    - is_ip: assess if provided string is a IP adress
    - nword: return the number of words
    - extension: return the extension of the site web (com, fr, org ...)
    - ndot: return the number of dots in the string
    - bothnumsandwords: assess if each word in the string is made with number and text (ex: 10aer)
    - synthetise: return a list with the output of the previous methods
    - listofwords: return the list of words in a string
    
    attributes:
    - string
    - words: created by listofwords() method
    '''
    
    def __init__(self, string):
        self.string = string
        
    def is_ip(self, verbose=True):
        import re
        result=all([i.isnumeric() for i in re.split(r"[.:/-]", self.string)])
        if verbose:
            print('is ip? {}'.format(result))
        return result
    
    def nword(self, verbose=True):
        import re
        result=len(re.split(r"[.:/-]", self.string))
        if verbose:
            print('n word is: {}'.format(result))
        return result
    
    def extension(self, verbose=True):
        import warnings
        if self.is_ip(verbose=False):
            warnings.warn('extension is not available for IP')
            return None
        else:
            result=self.string.rsplit('.')[-1]
            if verbose:
                print('extension is: {}'.format(result))
            return result
        
    def ndot(self, verbose=True):
        import re
        result=len(re.findall('\.', self.string))
        if verbose:
            print('n dot is: {}'.format(result))
        return result
    
    def bothnumsandwords(self, verbose=True):
        import re
        result=re.split(r"[.:/-]", self.string)
        regexp = re.compile(r'\D+\d+|\d+\D+')
        if any([regexp.search(res) for res in result]):
            result=1
        else:
            result=0            
   
        if verbose:
            print('contains both nums and words: {}'.format(result))
        
        return result
    
    def remove_www(self, verbose=True):
        import re
        if verbose:
            print('removing www.')
        result=re.sub(r"www\.","",self.string)
        return result        
        
    def synthetise(self, V=False):
        return [self.is_ip(verbose=V), self.nword(verbose=V), self.extension(verbose=V),self.ndot(verbose=V), self.bothnumsandwords(verbose=V)]
    
    def listofwords(self, verbose=True):
        import re        
        result=re.split('[.:/-]', self.string)
        if verbose:
            print('list of words is: {}'.format(result))
            
        self.words = result
        return result  
    
    
    
class WebSiteListAnalyser(StringAnalyzer):
    '''
    class for computing features on a list of website
    inherit from StringAnalyzer
    
    methods:
    - featuring: computes the features inherited from StringAnalyzer and return a pandas data frame
    - wordslist: return a pandas data frame with the list of words of each web site provided in the input website list
    '''
    def __init__(self, weblist):
        self.weblist = weblist
        
    def remove_all_www(self):
        print('removing www.')
        self.weblist=[StringAnalyzer(web).remove_www(verbose=False) for web in self.weblist]        

    def featuring(self):
        import pandas as pd
        import numpy as np       
       
        temp_df=np.array([StringAnalyzer(web).synthetise() for web in self.weblist])        
        columns = ['is_ip','nword','extension','ndot','bothnumsandwords']
        df = pd.DataFrame(temp_df, columns=columns)
        df['nword']=pd.to_numeric(df['nword'].values, downcast='integer')
        df['ndot']=pd.to_numeric(df['ndot'].values, downcast='integer')
        df['bothnumsandwords']=pd.to_numeric(df['bothnumsandwords'].values, downcast='integer')
        return df
     
    def wordslist(self):
        import pandas as pd
        import numpy as np       
       
        temp_df=np.array([StringAnalyzer(web).listofwords(verbose=False) for web in self.weblist])  
        columns = ['list_of_words']
        df = pd.DataFrame(temp_df, columns=columns)
        
        return df   

    
class MergeDFAndComputeFeature(WebSiteListAnalyser, StringAnalyzer):
    '''
     class for computing features and merging wiki dataframe with snippet data frame
    
    df1: data frame containing snippet and bing requests    
    df2: data frame containing  url and wiki info if available
   
    methods:
    - describedf: print a  description of a data frame
    - instantiate_df: instantiate the class MergeDF, with the 2 df that wil be concatenated
    - clean_adress: method for removing www. from address string before computing feature
    - mergedf: merge the provided data df and compute features on url adress list
    - nlp_flow:
    - nlp_preprocess:
    
    attributes:
    - df_merged: the merged data frame with the computed features
    
    '''
    
    def __init__(self, df1, df2):
        super()
        self.df1=df1.copy()
        self.df2=df2.copy()
        
    def __str__(self):
        import pandas as pd
        print("------------------------------------\n")
        self.describedf(self.df1,nco=self.df1.shape[1])
        print("------------------------------------\n")
        self.describedf(self.df2,nco=self.df2.shape[1])
        print("------------------------------------\n")
        return ""

    def describedf(self, df, nco):
        '''
        Small function useful for describing a dataframe
        '''
        import pandas as pd
        print('shape is nrow:{row} and ncol:{col}'.format(row=df.shape[0],col=df.shape[1]), '\n')
        print(df.head(), '\n')
        [print('type of col:{} is:{}'.format(df.columns[nc],df.iloc[:,nc].dtypes), '\n') for nc in range(0,nco)]
        
    def instantiate_df(self):
        self.wl1 = WebSiteListAnalyser(weblist=self.df1.url.values)
        self.wl2 = WebSiteListAnalyser(weblist=self.df2.url.values)
        
    def clean_adress(self):
#         self.wl1.remove_all_www()
        self.wl2.remove_all_www()
#         self.df1['url']=self.wl1.weblist
#         self.df2['url']=self.wl2.weblist
        
    def mergedf(self):
        import pandas as pd 
        
        if 'url' not in self.df1.columns or 'url' not in self.df2.columns:
            raise Exception("df1 and df2 should contain a column named 'url'")
        
        df_feat=self.wl2.featuring()
        df_concat=pd.concat([self.df2, df_feat], axis=1)
        
        if 'wiki_link' not in self.df2.columns:
            raise Exception("df2 should contain a column named 'wiki_link'")
                            
        df_concat.loc[df_concat.wiki_link.isnull(),'wiki']=0
        df_concat.loc[~df_concat.wiki_link.isnull(),'wiki']=1
        df_concat['list_of_words']=self.wl2.wordslist()
        self.df_merged = self.df1.merge(df_concat, left_on='url', right_on='url', suffixes=('_left', '_right'))
        
    
    
    def nlp_flow(self, text, nlp_en, nlp_fr, stopwords_en, stopwords_fr):
        # import spacy
        # from textblob import TextBlob
        import re
        import numpy as np

        
        # b = TextBlob(text)
        # lang = b.detect_language()
        # lang='en'
        text = text.lower()
        text=re.sub(r"///"," ",text)
        doc = nlp_en(text)
        lang = doc._.language['language']
        print('language is {}'.format(lang))
        
        if lang == 'en': 
            # Generate lemmatized tokens
            lemmas = [token.lemma_ for token in doc]
        
            # Remove stopwords and non-alphabetic tokens
            a_lemmas = [lemma for lemma in lemmas
                        if lemma.isalpha() and lemma not in stopwords_en]
        elif lang =='fr':

            doc = nlp_fr(text)
        
            # Generate lemmatized tokens
            lemmas = [token.lemma_ for token in doc]
        
            # Remove stopwords and non-alphabetic tokens
            a_lemmas = [lemma for lemma in lemmas
                        if lemma.isalpha() and lemma not in stopwords_fr]
        else:
            print('identified language is neither english nor french')
            a_lemmas = ['ERROR']
            # lang='UNKNOWN'                
    
        # print({'lem_words':' '.join(a_lemmas), 'language':lang})

        return {'lem_words':' '.join(a_lemmas), 'language':lang}
    
    def nlp_preprocess(self,stop_fr=None, stop_en=None):
        '''
        apply nlp_flow for cleaning and lemmatizing snippet on snippet feature
        
        parameters:
            - stop_fr & stop_en: allow to add author defined stopwords to the initial spacy stopwords list
       
        attributes:
            - columns 'lem_words' and 'language' added to self.df_merged        
        '''        
        import pandas as pd
        import spacy
        from spacy_langdetect import LanguageDetector
        import time
        
        nlp_en=spacy.load("en_core_web_md", disable=["tagger", "ner"])
        nlp_fr=spacy.load("fr_core_news_md", disable=["tagger", "ner"])
        nlp_en.add_pipe(LanguageDetector(), name='lang_detect', last=True)
        nlp_fr.add_pipe(LanguageDetector(), name='lang_detect', last=True)
        
        if stop_fr:
            if type(stop_fr)!=list:
                raise ValueError("stop_fr should be a list of strings")

            stopwords_fr=list(spacy.lang.fr.stop_words.STOP_WORDS)+stop_fr
            print('customized french stopwords list loaded')
        else:
            stopwords_fr=list(spacy.lang.fr.stop_words.STOP_WORDS)
            print('raw french stopwords list loaded')

        if stop_en:
            if type(stop_en)!=list:
                raise ValueError("stop_en should be a list of strings")
            stopwords_en=list(spacy.lang.en.stop_words.STOP_WORDS)+stop_en
            print('customized english stopwords list loaded')
        else:
            stopwords_en=list(spacy.lang.en.stop_words.STOP_WORDS)
            print('raw english stopwords list loaded')
        ti=1
        print("\n--- waiting {} seconds ---".format(ti))
        time.sleep(ti)
        # stopwords_en=spacy.lang.en.stop_words.STOP_WORDS
        # stopwords_fr=spacy.lang.fr.stop_words.STOP_WORDS
        # to do: add option for customize stop words
        
        preprocessed = self.df_merged.snippet.apply(lambda x: pd.Series(self.nlp_flow(x, nlp_en, nlp_fr, stopwords_en, stopwords_fr)))
        
        self.df_merged = pd.concat([self.df_merged, preprocessed], axis=1)

        
    
    def nlp_process(self,lang='fr',**kwargs):
        '''
        apply tf-idf transformation on lem_words feature
        
        parameters:
            -lang: 'fr' (default) or 'en'
            -kwargs:
                -analyzer
                -ngram_range
                -max_df
                -min_df
                
        Attributes:
            - tfidf
            - tfidf_features
        '''     
       
        from sklearn.feature_extraction.text import TfidfVectorizer    
        
        validM = {'fr':'French', 'en':'English'}
        if lang not in validM.keys():
            raise ValueError("lang parameter must be one of {}".format(list(validM.keys())))
        
        if 'lem_words' not in self.df_merged.columns:
            raise ValueError("please pre process the data frame for NLP first - see pre_process() method")
        
        # Kwargs
        Analyser=kwargs.get('analyzer', None)
        Ngram=kwargs.get('ngram_range', None)
        Maxdf=kwargs.get('max_df', None)
        Mindf=kwargs.get('min_df', None)
        
        # if kwargs not provided, set to default values
        if not Analyser:
            Analyser='word'
        else:
            print('analyser is set to: {}'.format(Analyser))
        if not Ngram:
            Ngram=(1,1)
        else:
            print('ngram_range is set to: {}'.format(Ngram))
        if not Maxdf:
            Maxdf=1.0
        else:
            print('max_df is set to: {}'.format(Maxdf))
        if not Mindf:
            Mindf=1
        else:
            Mindf=int(Mindf)
            print('min_df is set to: {}'.format(Mindf))
        
        # instantiate tf_idf
        tfidf_vectorizer = TfidfVectorizer(analyzer=Analyser,ngram_range=Ngram,max_df=Maxdf, min_df=Mindf)
        
        print('chosen language is {}'.format(validM[lang]))
        if lang=='fr':
            df=self.df_merged[self.df_merged['language']=='fr']
        else:
            df=self.df_merged[self.df_merged['language']=='en']
           
        tfidf=tfidf_vectorizer.fit_transform(df.lem_words.values)
        
        self.lang=lang
        self.tfidf = tfidf.A
        self.tfidf_features = tfidf_vectorizer.get_feature_names()
        print('tfidf computed, see tfidf & tfidf_features attributes')
        
        
    def prepare_corpus(self,lang='fr'):
        """
        Input  : clean document
        Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
        Output : term dictionary and Document Term Matrix
        https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
        """
        # import os.path
        from gensim import corpora
        # from gensim.models import LsiModel
        # from nltk.tokenize import RegexpTokenizer
        # from nltk.corpus import stopwords
        # from nltk.stem.porter import PorterStemmer
        # from gensim.models.coherencemodel import CoherenceModel
        # import matplotlib.pyplot as plt
        
        validM = {'fr':'French', 'en':'English'}
        if lang not in validM.keys():
            raise ValueError("lang parameter must be one of {}".format(list(validM.keys())))
        
        if 'lem_words' not in self.df_merged.columns:
            raise ValueError("please pre process the data frame for NLP first - see pre_process() method")
        
        print('chosen language is {}'.format(validM[lang]))
        if lang=='fr':
            df=self.df_merged[self.df_merged['language']=='fr']
        else:
            df=self.df_merged[self.df_merged['language']=='en']
        
        doc_clean=[]
        [doc_clean.append(i.split()) for i in df.lem_words.values]
        
        
        # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
        dictionary = corpora.Dictionary(doc_clean)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        # generate LDA model
        
        self.doc_clean=doc_clean
        self.dictionary=dictionary
        self.doc_term_matrix=doc_term_matrix
        
        return dictionary,doc_term_matrix
    
    def create_gensim_lsa_model(self, number_of_topics,words,lang='fr'):
        """
        Input  : clean document, number of topics and number of words associated with each topic
        Purpose: create LSA model using gensim
        Output : return LSA model
        """
        from gensim.models import LsiModel

        dictionary,doc_term_matrix=self.prepare_corpus(lang=lang)
        # generate LSA model
        lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        print('\n',lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
        
        self.lsamodel=lsamodel
        self.number_of_topics_lsa=number_of_topics
        return lsamodel
    
    def compute_coherence_values(self, dictionary, doc_term_matrix, lang, start=2, stop=10,step=1):
        """
        Input   : dictionary : Gensim dictionary
                  corpus : Gensim corpus
                  texts : List of input texts
                  stop : Max num of topics
        purpose : Compute c_v coherence for various number of topics
        Output  : model_list : List of LSA topic models
                  coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        # print(lang)
        from gensim.models import LsiModel
        from gensim.models.coherencemodel import CoherenceModel
       
        validM = {'fr':'French', 'en':'English'}
        if lang not in validM.keys():
            raise ValueError("lang parameter must be one of {}".format(list(validM.keys())))
        
        if 'lem_words' not in self.df_merged.columns:
            raise ValueError("please pre process the data frame for NLP first - see pre_process() method")
        
        print('chosen language is {}'.format(validM[lang]))
        if lang=='fr':
            df=self.df_merged[self.df_merged['language']=='fr']
        else:
            df=self.df_merged[self.df_merged['language']=='en']
            
        doc_clean=[]
        [doc_clean.append(i.split()) for i in df.lem_words.values]
        
        coherence_values = []
        model_list = []    
        
        for num_topics in range(start, stop, step):
            print('\nnumber of topics is: {}'.format(num_topics))
            # generate LSA model
            model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values    
    

    def extract_words_by_lsa_topic(self, num_words=10):
        '''
        print top words associated with each topic for lsa
        '''
        import numpy as np
        from gensim.models import LsiModel
        
        words_array_lsa=[]
        
        for i,j in self.lsamodel.show_topics(num_topics=self.number_of_topics_lsa, num_words=10, formatted=False):
            for k in range(0,num_words):
                words_array_lsa.append(j[k][0])
        
        words_array_lsa=np.array(words_array_lsa).reshape(self.number_of_topics_lsa,-1)
        self.words_array_lsa=words_array_lsa
        print(words_array_lsa)



    def create_gensim_lda_model(self, number_of_topics,words,lang='fr'):
        """
        Input  : clean document, number of topics and number of words associated with each topic
        Purpose: create LDA model using gensim
        Output : return LDA model
        """
        from gensim.models import LdaModel, LdaMulticore

        dictionary,doc_term_matrix=self.prepare_corpus(lang=lang)
        # generate LDA model
        ldamodel = LdaModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        print('\n',ldamodel.print_topics(num_topics=number_of_topics, num_words=words))
        
        self.ldamodel=ldamodel
        self.number_of_topics_lda=number_of_topics

        return ldamodel
    
    
    def extract_words_by_lda_topic(self, num_words=10):
        '''
        print top words associated with each topic for lda
        '''
        import numpy as np

        
        words_array_lda=[]
        
        for i,j in self.ldamodel.show_topics(num_topics=self.number_of_topics_lda, num_words=10, formatted=False):
            for k in range(0,num_words):
                words_array_lda.append(j[k][0])
        
        words_array_lda=np.array(words_array_lda).reshape(self.number_of_topics_lda,-1)
        self.words_array_lda=words_array_lda
        print(words_array_lda)

    
    def lda_predicted_cluster(self, texts=None):
        '''
        predict cluster of corpus based on LDA model
        return pandas data frame with 2 columns
        - cluster (most likely)
        - proba
        
        https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/topic_methods.ipynb
        
        bug fixed: use itemgetter to find the max proba
        
        to do:
            check if results are consistant with langage selection
        '''
        from operator import itemgetter
        import pandas as pd
        
        if not texts:
            all_topics = self.ldamodel.get_document_topics(self.doc_term_matrix, per_word_topics=True)
               
        else:
            # NEED TO TEST FUNCTIONNALITY FOR HIDDEN BUGS
            texts=pd.DataFrame({'num':range(0,len(texts)),'text':texts})
            # print(texts)
            texts=self.nlp_preprocess_new_texts(texts)
            # print(texts)
            final_texts=[]
            [final_texts.append(i.split()) for i in texts.lem_words.values]
            doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in final_texts]
            all_topics = self.ldamodel.get_document_topics(doc_term_matrix, per_word_topics=True)

        lda_cluster=[]    
        [lda_cluster.append([max(doc_topics,key=itemgetter(1))[0],max(doc_topics,key=itemgetter(1))[1]]) for doc_topics, _, _ in all_topics]

        lda_cluster=pd.DataFrame(lda_cluster, columns=['cluster','proba'])
        
        self.lda_cluster=lda_cluster



    def nlp_preprocess_new_texts(self, texts,stop_fr=None, stop_en=None):
        import pandas as pd
        import spacy
        from spacy_langdetect import LanguageDetector
        import time
        
        nlp_en=spacy.load("en_core_web_md", disable=["tagger", "ner"])
        nlp_fr=spacy.load("fr_core_news_md", disable=["tagger", "ner"])
        nlp_en.add_pipe(LanguageDetector(), name='lang_detect', last=True)
        nlp_fr.add_pipe(LanguageDetector(), name='lang_detect', last=True)
        
        if stop_fr:
            if type(stop_fr)!=list:
                raise ValueError("stop_fr should be a list of strings")

            stopwords_fr=list(spacy.lang.fr.stop_words.STOP_WORDS)+stop_fr
            print('customized french stopwords list loaded')
        else:
            stopwords_fr=list(spacy.lang.fr.stop_words.STOP_WORDS)
            print('raw french stopwords list loaded')

        if stop_en:
            if type(stop_en)!=list:
                raise ValueError("stop_en should be a list of strings")
            stopwords_en=list(spacy.lang.en.stop_words.STOP_WORDS)+stop_en
            print('customized english stopwords list loaded')
        else:
            stopwords_en=list(spacy.lang.en.stop_words.STOP_WORDS)
            print('raw english stopwords list loaded')
        
        # print(texts)
        # print(texts.columns)
        # print(texts.text)
        preprocessed = texts.text.apply(lambda x: pd.Series(self.nlp_flow(x, nlp_en, nlp_fr, stopwords_en, stopwords_fr)))
        
        return preprocessed

    def plot_graph(self, start, stop, step, lang='fr'):
        '''
        estimate optimal number of topics for lsa
        '''

       
        import matplotlib.pyplot as plt        
  
       
    
        dictionary, doc_term_matrix=self.prepare_corpus(lang=lang)
        model_list, coherence_values = self.compute_coherence_values(dictionary, doc_term_matrix, lang,
                                                                     start, stop, step)
        x = range(start, stop, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()
        
    
            
    def preparedataset(self, add_user_feature=True):
        import pandas as pd
        import numpy as np
        
        if self.df_merged.empty:              
            raise ValueError('df_merged is missing, please use .mergedf() method first')
        
        try:
            np.any(self.tfidf)
        except:                          
            raise ValueError('tfidf is missing, please use .tfidf() method first')
            
        if add_user_feature:      
            self.dfX = pd.concat([self.df_merged.loc[self.df_merged['language']==self.lang,
                                        ['ndot','bothnumsandwords']].reset_index(drop=True),
                     pd.DataFrame(self.tfidf,columns=self.tfidf_features)],axis=1)
            print('\ndata set prepared including features: \n -ndot \n -bothnumsandwords ')
            print('\nShape is: \n -nrow: {} \n -ncol: {}'.format(self.dfX.shape[0], self.dfX.shape[1]))

        else:
            self.dfX=pd.DataFrame(self.tfidf,columns=self.tfidf_features)
            print('\ndata set prepared with only: \n -tfidf features')
            print('\nShape is: \n -nrow: {} \n -ncol: {}'.format(self.dfX.shape[0], self.dfX.shape[1]))

                 
        self.dfy = self.df_merged.loc[self.df_merged['language']==self.lang,['wiki']].reset_index(drop=True)    
        print('\n ------ data set ready ------\n')

                   
        

        
        
        
        
        
        