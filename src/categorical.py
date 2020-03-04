from sklearn import preprocessing

"""
Label Encoding
One Hot Encoding
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

    def _ohe(self):

        self.output_df = pd.get_dummies(self.df, columns=self.cat_feats)    
        return self.output_df


    def fit_transform(self):

        if self.enc_type == 'label':
            return self._label_encoder()

        elif self.enc_type == 'ohe':
            return self._ohe()

        else:
            raise Exception('Encoding Type not understood')

    #def transform(self,dataframe):

        #if self.enc_type == 'label':
          #       dataframe.loc[:col] = self.label_encoder[col].transform(dataframe[col].values.tolist())

        #elif self.enc_type == 'ohe':
            


    
if __name__ == '__main__':

    import pandas as pd 
    train = pd.read_csv('./input/train_cat.csv').head(100)
    test = pd.read_csv('./input/test_cat.csv').head(50)
    test['target']=-1

    train_idx = train['id'].values
    test_idx  = test['id'].values

    full_data = pd.concat([train,test],ignore_index=True)

    cols = [col for col in full_data.columns if col not in ['id','target']]
    print(cols)

    cat_feats = CategoricalFeatures(full_data,
                                    categorical_features = cols,
                                    encoding_type =  'ohe',
                                    handle_na = True)  

    full_data_transformed = cat_feats.fit_transform()
    
    train_transformed = full_data_transformed[full_data_transformed['id'].isin(train_idx)]
    test_transformed = full_data_transformed[full_data_transformed['id'].isin(test_idx)]

    print(train_transformed.shape)
    print(test_transformed.shape)
    





