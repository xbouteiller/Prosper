def describedf(df,nco=1):
    '''
    Small function useful for describing a dataframe
    '''
    print('shape is nrow:{row} and ncol:{col}'.format(row=df.shape[0],col=df.shape[1]), '\n')
    print(df.head(), '\n')
    [print('type of col:{} is:{}'.format(df.columns[nc],df.iloc[:,nc].dtypes), '\n') for nc in range(0,nco)]
    

class StringAnalyzer():
    
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
        result=any([regexp.search(res) for res in result]) 
   
        if verbose:
            print('contains both nums and words: {}'.format(result))
        
        return result        
        
        
    def synthetise(self, V=False):
        return [self.is_ip(verbose=V), self.nword(verbose=V), self.extension(verbose=V),self.ndot(verbose=V), self.bothnumsandwords(verbose=V)]
    
    def listofwords(self, verbose=True):
        import re        
        result=re.split('[.:/-]', self.string)
        if verbose:
            print('list of words is: {}'.format(result))
        return result  
    
    
    
class WebSiteListAnalyser(StringAnalyzer):
    
    def __init__(self, weblist):
        self.weblist = weblist

    def featuring(self):
        import pandas as pd
        import numpy as np       
       
        temp_df=np.array([StringAnalyzer(web).synthetise() for web in self.weblist])        
        columns = ['is_ip','nword','extension','ndot','bothnumsandwords']
        df = pd.DataFrame(temp_df, columns=columns)
        df['nword']=pd.to_numeric(df['nword'].values, downcast='integer')
        df['ndot']=pd.to_numeric(df['ndot'].values, downcast='integer')
        return df
     
    def wordslist(self):
        import pandas as pd
        import numpy as np       
       
        temp_df=np.array([StringAnalyzer(web).listofwords(verbose=False) for web in self.weblist])  
        columns = ['list_of_words']
        df = pd.DataFrame(temp_df, columns=columns)
        
        return df
    
        
            