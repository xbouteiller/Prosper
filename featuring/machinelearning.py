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
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dfX,
                                                                                self.dfy,
                                                                                random_state=random_state,
                                                                                test_size=test_size)
        if stratify:
            print('\n data were splitted with parameters: \n -test_size: {} \n -stratify: {}'.format(test_size, stratify.columns))
        
        # return self.X_train, self.X_test, self.y_train, self.y_test
        
    def instantiate_classif(self, classifier='lr', **kwargs):        
        
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
        from sklearn.metrics import confusion_matrix
        try:            
            self.classif.fit(self.X_train, self.y_train.values.ravel())
        except:
            raise ValueError('Data set are missing, please use split_data() method first')
            
        self.score = self.classif.score(self.X_test, self.y_test.values.ravel())
        print('\n Score is {0:.3f}'.format(self.score))
        self.confusion_matrix = confusion_matrix( self.y_test.values.ravel(), self.classif.predict(self.X_test))
        print('Confusion matrix is \n',self.confusion_matrix)
        
    def do_clustering(self, eps=0.2, min_samples=5, metric='euclidean'):
        from sklearn.cluster import DBSCAN        
        db=DBSCAN(eps=eps, min_samples=min_samples, metric=metric )
        self.dfy_db = db.fit_predict(self.dfX)
        print('\n ------ dbscan clustering done ------\n')

    
                                        