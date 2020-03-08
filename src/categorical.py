from sklearn import preprocessing
import category_encoders as ce

"""

Classical Encoding
-----------------
Label Encoding
One Hot Encoding
Binary Encoding
BaseN Encoding

Bayesian Encoding
-----------------
Target Encoding 
Weight of Evidence
Leave one out
James-stein
M-estimator

"""

class CategoricalFeatures:

    def __init__(self,df, categorical_features, encoding_type, handle_na):
        '''
        df : pandas dataframe
        categorical_features : list of columns 
        encoding_type : label, binary, ohe 
        '''

        self.df  = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoder = {}
        self.target_encoder = {}

        if handle_na:
            for col in self.cat_feats:
                self.df.loc[:,col] = self.df.loc[:,col].fillna(-99999).astype('str') 

        self.output_df = df.copy(deep = True)

    def _label_encoder(self):

        for col in self.cat_feats:
            lbl =    preprocessing.LabelEncoder()
            lbl.fit_transform(self.df[col].values.tolist())
            self.output_df.loc[:,col] = lbl.transform(self.df[col].values.tolist())
            self.label_encoder[col] = lbl

        return self.output_df
    
    def _target_encoder(self, smoothing =0.2):
        
        '''
        Bayesian Encoding
        '''
        
        for col in self.cat_feats:
            ce_target =  ce.TargetEncoder(cols = col, smoothing = smoothing)        
            ce_target.fit(self.df[col], self.df['target'])           
            self.output_df.loc[:,col]= ce_target.transform(self.df[col])
            self.target_encoder[col] = ce_target
     
        return self.output_df
      
    def _ohe(self):

        self.output_df = pd.get_dummies(self.df, columns=self.cat_feats)    
        return self.output_df


    def fit_transform(self):

        if self.enc_type == 'label':
            return self._label_encoder()

        elif self.enc_type == 'ohe':
            return self._ohe()
        
        elif self.enc_type == 'target':
            return self._target_encoder()

        else:
            raise Exception('Encoding Type not understood')
            
    def transform(self,dataframe):
        
        for col in self.cat_feats:
            ce_target =  self.target_encoder[col]        
            dataframe.loc[:,col]= ce_target.transform(dataframe[col])
           
        return dataframe
      
    


    
if __name__ == '__main__':

    import pandas as pd 
    train = pd.read_csv('./input/train_cat.csv')
    test = pd.read_csv('./input/test_cat.csv')
    test['target']=-1
    sample = pd.read_csv('./input/sample_submission_cat.csv')
    train_idx = train['id'].values
    test_idx  = test['id'].values

    full_data = pd.concat([train,test],ignore_index=True)
    full_data = train
    cols = full_data.columns.tolist()
    cols = [col for col in full_data.columns if col not in ['id','target'] ]
    unique = dict()
    for col in cols:
        unique[col] = len(full_data[col].unique().tolist())

    cols_keep  = [col for col in cols if unique[col] < 20]
    
    nominal_col = ['nom_5','nom_6','nom_7','nom_8','nom_9','']
    
    
    ord_1_mapping = {'Novice': 160597,
                     'Expert':1,
                     'Contributor':    109821,
                     'Grandmaster':2,
                     'Master':   95866
    }                 
    
    ord_2_mapping = {'Freezing':       142726,
                     'Warm':           124239,
                     'Cold':            97822,
                     'Boiling Hot':     84790,
                     'Hot':             67508,
                     'Lava Hot':        64840  }                 
    


   # col_nominal = [col for col in full_data.columns if col.startswith('nom')]
    
    cols_ord = 


'''
 to do 
 1. treat ordinal value seperately
             - label encoder
             - replacing with event rate

 2. treat missing value seperately
             - replace with most likely value wrt event rate

 '''
 
    cat_feats = CategoricalFeatures(full_data_transformed,
                                    categorical_features = cols,
                                    encoding_type =  'label',
                                    handle_na = True)  

    full_data_transformed = cat_feats.fit_transform()
    
    train_1 = target_enc.df 
    
    
    
    target_enc = CategoricalFeatures(full_data,
                                    categorical_features = cols,
                                    encoding_type =  'target',
                                    handle_na = True)  
    
    target_enc_test = CategoricalFeatures(test,
                                    categorical_features = cols,
                                    encoding_type =  'target',
                                    handle_na = True)  
    
    

   
    train_transformed = target_enc.fit_transform()
    test_transformed = target_enc.transform(test)
    
    unique = dict()
    for col in cols:
        unique[col] = list(set(test[col]))
    
    train_transformed['day'] = train['day'] 
    train_transformed['month'] = train['month'] 
    
    target_enc.cat_feats
            for col in target_enc.cat_feats:
                test.loc[:,col] = test.loc[:,col].fillna(-99999).astype('str') 

    dataframe = test
  
    col ='bin_0'
    for col in target_enc.cat_feats:
        ce_target =  target_enc.target_encoder[col]
        dataframe.loc[:,col]= ce_target.transform(test.loc[:,col])
        train_1.loc[:,col]= ce_target.transform(train.loc[:,col])
        test.loc[:,col]= ce_target.transform(test.loc[:,col])

    test_transformed = target_enc.transform(test)
    test_transformed['day'] = test['day'] 
    test_transformed['month'] = test['month'] 
    
    
    test_transformed['id'] = test['id']

    train_transformed = full_data_transformed[full_data_transformed['id'].isin(train_idx)]
    test_transformed = full_data_transformed[full_data_transformed['id'].isin(test_idx)]

    train_transformed = full_data_transformed
    unique_val = dict()
    for col in cols:
        unique_val[col] = train_transformed[col].unique().tolist()

    cols = [col for col in full_data_transformed.columns if col not in ['id','target']]    
    
    print(cols)

    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression()
    logreg.fit(train_transformed[cols], train.target.values)

    preds = logreg.predict_proba(test_transformed[cols])[:,1]
    

    from sklearn import ensemble
    
    MODELS = {
        'randomforest' : ensemble.RandomForestClassifier(n_estimators = 200, n_jobs=-1,verbose=2),
        'extratrees' : ensemble.ExtraTreesClassifier(n_estimators = 200,n_jobs=-1,verbose =2)
        'logistic' : LogisticRegression()
    }

    logreg = MODELS['randomforest']
    logreg.fit(train_transformed[cols], train.target.values)

    preds = logreg.predict_proba(test_transformed[cols])[:,1]
    sample.loc[:,'target'] = preds
    sample['id'] = sample['id'].astype('int')
    sample.to_csv('./models/submission_target_encoding.csv',index= False)
    
    
    