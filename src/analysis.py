import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
import seaborn as sns

class Analysis(object):
    '''
    '''
    def __init__(self, num_cols=None, target_var=None, threshold=None):
        self.num_cols_ = num_cols
        self.target_var_ = target_var
        self.data_profile_ = None
        self.save_path_ = None
    
    def _describe_numeric_cols(self, df):
        dic = {}
        for col in self.num_cols_:
            dic[col] = {}
            dic[col] ['Fill Rate'] = 1 - df[col].isnull().sum()/(df.shape[0])
            dic[col] ['Mean'] = df[col].min()
            dic[col] ['Std_Dev'] = df[col].std(skipna=True)
            dic[col] ['Min'] = df[col].min()
            dic[col] ['25%'] = df[col].quantile([.25]).values[0] 
            dic[col] ['50%'] = df[col].quantile([.50]).values[0] 
            dic[col] ['75%'] = df[col].quantile([.75]).values[0] 
            dic[col] ['99%'] = df[col].quantile([.99]).values[0] 
            dic[col] ['Max'] = df[col].max()
        return dic
    
    
    def _describe_categorical_cols(self, df):
        dic = {}
        for col in self.cat_cols_:
            dic[col] = {}
            dic[col] ['Fill Rate'] = 1 - df[col].isnull().sum()/(df.shape[0])
            dic[col] ['Categories'] = df[col].min()
            dic[col] ['Std_Dev'] = df[col].std(skipna=True)
            dic[col] ['Min'] = df[col].min()
            dic[col] ['25%'] = df[col].quantile([.25]).values[0] 
            dic[col] ['50%'] = df[col].quantile([.50]).values[0] 
            dic[col] ['75%'] = df[col].quantile([.75]).values[0] 
            dic[col] ['99%'] = df[col].quantile([.99]).values[0] 
            dic[col] ['Max'] = df[col].max()
        return dic
    
    def generate_univariate(self, df, num_cols=[], 
                            target_var='target'):
       
        if num_cols:
            self.num_cols_ = num_cols
        if target_var:
            self.target_var_ = target_var
       
        cols_order =['Fill Rate','Mean','Std_Dev','Min','25%', '50%', '75%', '99%', 'Max']
        numerical_feature_analyis = self._describe_numeric_cols(df, cols=cols).T[cols_order]
    
        self.data_profile_ =  [ numerical_feature_analyis]
        self.save_path_ = '../output2/univariate_analysis.xlsx'
        
    def generate_univariate_categorical(self,df, cat_cols=[],
                                        target_var = 'target'):
        
        if num_cols:
            self.num_cols_ = cat_cols
        if target_var:
            self.target_var_ = target_var
       
        cols_order =['Fill Rate','Mean','Std_Dev','Min','25%', '50%', '75%', '99%', 'Max']
        numerical_feature_analyis = self._describe_numeric_cols(df, cols=cols).T[cols_order]
    
        self.data_profile_ =  [ numerical_feature_analyis]
        self.save_path_ = '../output2/univariate_analysis.xlsx'
        
        
    def save_report(self, save_path=Nsone):
        if save_path:
            self.save_path_ = save_path
        tab2 = self.data_profile_
        writer = pd.ExcelWriter(save_path)
        tab2.to_excel(writer, 'Numerical Variables Details', index=False)
      
        print('Univariate Analysis Saved!')

a = Analysis()
a.generate_univariate(data,num_cols = [])


import pandas as pd
