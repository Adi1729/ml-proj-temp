import pandas as pd 
from sklearn import preprocessing
import os
from sklearn import ensemble
from sklearn import metrics
from sklearn import ensemble
from . import dispatcher
import joblib
import numpy as np

TEST_DATA= os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():

    test_df = pd.read_csv(TEST_DATA) 
    
    test_idx = test_df['id'].values
    cols = joblib.load(f"models/columns.pkl")
    predictions =[]
    
    for FOLD in range(5):
            
        encoders = joblib.load(f"models/{MODEL}_{FOLD}_label_encoder.pkl")
        print(f'Loading {MODEL}_{FOLD}...')
        model =  joblib.load(f"models/{MODEL}_{FOLD}.pkl")
        test_df = pd.read_csv(TEST_DATA)
    
        for c in cols:
            lbl =   encoders[c]
            test_df.loc[:,c] = lbl.transform(test_df[c].values.tolist())
        
        # data is ready to train
        clf = model
        test_df = test_df[cols]
        preds = clf.predict_proba(test_df)[:,1]
        
        if FOLD==0:
            predictions = preds
        else:
            predictions += preds
 
    predictions /=5

    return pd.DataFrame(np.column_stack((test_idx,preds)),columns= ['id','target'])


if __name__ == '__main__':

    sub = predict()
    sub.to_csv(f'models/submission_{MODEL}.csv',index= False)




