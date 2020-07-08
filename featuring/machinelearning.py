# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:50:37 2020

@author: xavier
"""

  



class MachineLearning():
    
    import pandas as pd
    
    def __init__(self, dfX, dfy):
        if dfX.shape[0]!=dfy.shape[0]:
            raise ValueError("dfX and dfy should have the same rows number")
        if dfy.shape[1]>1:
            raise ValueError("dfy should have only one column")
        if len(dfX.select_dtypes(exclude=['number']).columns)>0:
            raise ValueError("dfX should contain only numeric columns")       
        self.dfX=dfX       
        self.dfy=dfy
        
    def split_data(self, random_state=99, test_size=0.5, stratify=None):
        '''
        split dfX & dfy to training and testing set
        '''

      
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dfX,
                                                                                self.dfy,
                                                                                random_state=random_state,
                                                                                test_size=test_size)
        if stratify:
            print('\n data were splitted with parameters: \n -test_size: {} \n -stratify: {}'.format(test_size, stratify.columns))
        
        # return self.X_train, self.X_test, self.y_train, self.y_test
        
    def instantiate_classif(self, classifier='lr', **kwargs):        
        '''
        Instantiate a scikit classifier among:
            - random forest
            - naive bayes
            - logistic regression
            
        main hyperparameters are customizable via **kwargs arguments
        
        try to predict if the website has a wiki page
        
        '''
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
         
        #check if the classifier is allowed
        validClassif = {'rf':'random forest',
                        'nb': 'naive Bayes', 
                        'lr': 'logistic regression'}
        if classifier not in validClassif:
            raise ValueError("results: classifier must be one of {}".format(list(validClassif.keys())))               
            
        # kwargs : hyperparameters for classifier
        max_depth=kwargs.get('max_depth', None)
        class_weight=kwargs.get('class_weight', None)
        n_estimators=kwargs.get('n_estimators', None)
        penalty=kwargs.get('penalty', None)
        C=kwargs.get('C', None)
        solver=kwargs.get('solver', None)
        random_state=kwargs.get('random_state', None)
        
        if not max_depth:
            max_depth=6
        if not class_weight:
            class_weight='balanced_subsample' 
        if not n_estimators:
            n_estimators=500
        if not penalty:
           penalty='none'
        if not C:
           C=1
        if not solver:
           solver='lbfgs'
        if not random_state:
           random_state=99
        
    
        print('\n Classifier is {} '.format(validClassif[classifier]))
        
        # initialize classifier
        if classifier == 'rf':
            print('\n hyperparameters are: \n -n_estimators: {}\n -max_depth: {}\n -class_weight: {}'.format(n_estimators,max_depth,class_weight))
            self.classif = RandomForestClassifier(n_estimators=n_estimators,
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  class_weight=class_weight)
            
        if classifier == 'nb':
            print('\n hyperparameters are: \n -{a}'.format(a=None))
            self.classif = GaussianNB()
            
        if classifier == 'lr':
            print('\n hyperparameters are: \n -C: {}\n -penalty: {}\n -solver: {}'.format(C,penalty,solver))
            self.classif = LogisticRegression(random_state=random_state,
                                              penalty=penalty,
                                              C=C,
                                              solver=solver)
            
        # return self.classif
            
    def fit_classif(self):
        '''
        fit the instantiated classifier then return score and confusion matrix
        '''        
        from sklearn.metrics import confusion_matrix
        try:            
            self.classif.fit(self.X_train, self.y_train.values.ravel())
        except:
            raise ValueError('Data set are missing, please use split_data() method first')
            
        self.score = self.classif.score(self.X_test, self.y_test.values.ravel())
        print('\n Score is {0:.3f}'.format(self.score))
        self.confusion_matrix = confusion_matrix( self.y_test.values.ravel(), self.classif.predict(self.X_test))
        print('Confusion matrix is \n',self.confusion_matrix)
        
    def do_dbscan(self, eps=0.2, min_samples=5, metric='euclidean'):
        '''
        fit a dbscan clustering then return prediction
        '''
        from sklearn.cluster import DBSCAN        
        db=DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-2 )
        self.dfy_db = db.fit_predict(self.dfX)
        print('\n ------ dbscan clustering done ------\n')


    def do_pham(self, max_k = 10):
        '''
        https://github.com/Vonatzki/pham_dimov_python/blob/master/Pham-Dimov%20Python%20Implementation.ipynb
        '''
        # Pertinent modules for this proof
        from sklearn.cluster import KMeans        
        import pandas as pd
        from pandas import DataFrame

        rng = range(1, max_k + 1)
        
        sks = pd.Series(index = rng)
        As = pd.Series(index = rng)
        fks = pd.Series(index = rng)
        
        nd = self.dfX.shape[1]
        
        print("Number of dimensions detected: %s\n" % nd)
        
        pham_output = DataFrame()
        
        for k in rng:
            model = KMeans(n_clusters = k)
            model = model.fit(self.dfX)
            
            # Compute for the Sk
            sk = model.inertia_
            sks[k] = sk
            
            # Compute for the alpha
            if (k == 2) & (nd > 1):
                a = 1 - (3 / (4 * float(nd)))
            elif (k > 2) & (nd > 1):
                a = As[k-1] + ((1-As[k-1])/6)
            else:
                a = None
                
            As[k] = a
            
            # Compute f(k)
            if k == 1:
                fk = 1.0
            elif (sks[k-1] != 0):
                fk = sk / (a * sks[k-1])
            elif (sks[k-1] == 0):
                fk = 1.0
            fks[k] = fk
            
            print("CENTROID %s || sk: %s\tfk: %s\ta: %s" % (k, sk, fk, a))
            
            output = DataFrame({"K":[k],"Sk":[sk], "F(k)":[fk], "ALPHA":[a]})
            
            pham_output = pd.concat([pham_output, output], axis = 0)
            
            
        self.dfy_pham = model.predict(self.dfX)
            
            
                                            