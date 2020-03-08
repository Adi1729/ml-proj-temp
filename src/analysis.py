import pandas as pd
import numpy as np
import os
<<<<<<< HEAD
from matplotlib import pyplot as plt
import seaborn as sns

class Analysis:
    '''
    Gives excel with 2 tabs : 
        univariate analyis  of numberical columns
        bivariate analysis of categorical columns
    '''

    def __init__(self, df, num_cols=None, cat_cols=None, target_var=None):

        self.df = df
=======

from matplotlib import pyplot as plt
import seaborn as sns

class Analysis(object):
    '''
    '''
    def __init__(self, num_cols=None, target_var=None, threshold=None):
>>>>>>> 2fb3fc6d58974c5abf5e33b64037677679bc4976
        self.num_cols_ = num_cols
        self.target_var_ = target_var
        self.data_profile_ = None
        self.save_path_ = None
<<<<<<< HEAD
                
        if cat_cols:
            self.cat_cols_ = cat_cols
        else:
            self.cat_cols_ = [col for col in self.df.columns.tolist() if col not in self.num_cols_ + [self.target_var_]]
        


    def _describe_numeric_cols(self):
        
        '''
        df : dataframe

        Return:
        ----------
        dic : dictionary of numerical columns with basic stats
        '''
        
        dic = {}
        for col in self.num_cols_:
            dic[col] = {}
            dic[col] ['Fill Rate'] = 1 - self.df[col].isnull().sum()/(self.df.shape[0])
            dic[col] ['Mean'] = self.df[col].min()
            dic[col] ['Std_Dev'] = self.df[col].std(skipna=True)
            dic[col] ['Min'] = self.df[col].min()
            dic[col] ['25%'] = self.df[col].quantile([.25]).values[0] 
            dic[col] ['50%'] = self.df[col].quantile([.50]).values[0] 
            dic[col] ['75%'] = self.df[col].quantile([.75]).values[0] 
            dic[col] ['99%'] = self.df[col].quantile([.99]).values[0] 
            dic[col] ['Max'] = self.df[col].max()
        

        return dic
    
    
    def _describe_categorical_cols(self):


        '''
        df : dataframe

        Return:
        ----------
        main_stats : dictionary of categorical columns with univariate and bivariate stats

        '''

        dic = {}
        main_stats = pd.DataFrame(columns = ['Features','Levels','Class count','Event count','Non-Event count'])
        
        fill_rate = (100-self.df.isnull().sum()*100/self.df.shape[0]).reset_index()
        fill_rate.columns = ['Features','Fill_rate']
        
        for col in self.cat_cols_:

            data_filter = self.df.dropna(subset = [col])
            elements = data_filter[col].unique().tolist()

            for element in elements:
                stats ={}

                stats['Features'] = col
                stats['Levels'] = element
                stats['Class count'] = data_filter[data_filter[col]==element].shape[0]
                stats['Event count'] = data_filter[data_filter[col]==element].loc[data_filter[self.target_var_]==1,].shape[0]
                stats['Non-Event count'] = data_filter[data_filter[col]==element].loc[data_filter[self.target_var_]==0,].shape[0]

                main_stats =main_stats.append(stats,ignore_index = True)
        
        main_stats['Class Distribution'] = main_stats['Class count']*100/self.df.shape[0]
        main_stats['Event Rate'] = main_stats['Event count']*100/main_stats['Class count']
        main_stats['Non-Event Rate'] = main_stats['Non-Event count']*100/main_stats['Class count']
        
        return fill_rate.merge(main_stats,how='inner')
    
    def generate_report(self):  
       
        cols_order =['Fill Rate','Mean','Std_Dev','Min','25%', '50%', '75%', '99%', 'Max']
        numerical_feature_analyis = pd.DataFrame(self._describe_numeric_cols()).T[cols_order]
        categorical_bivariate_analyis = self._describe_categorical_cols()
        self.data_profile_ =  [numerical_feature_analyis,categorical_bivariate_analyis]
        
        
    def save_report(self, save_path=None):

        if save_path:
            self.save_path_ = save_path

        self.generate_report()
        tab1,tab2 = self.data_profile_
        writer = pd.ExcelWriter(self.save_path_)
        tab1.to_excel(writer, 'Numerical Variables Details', index=False)
        tab2.to_excel(writer, 'Categorical Variable Details', index=False)

        writer.save()
        print('Analysis Report Saved!')


if '__name__' == '__main__':

    import pandas as pd 
    train = pd.read_csv('./input/train_cat.csv')
    num_cols =['id']
    cat_cols = [col for col in train.columns.tolist() if col not in ['id','target']]
    Analys = Analysis(df = train,
                        num_cols = ['id'],
                        cat_cols = cat_cols,
                        target_var ='target')
    Analys.save_report(save_path = './output/analysis.xlsx')
    
    
=======
    
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
>>>>>>> 2fb3fc6d58974c5abf5e33b64037677679bc4976
