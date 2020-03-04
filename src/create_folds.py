import pandas as pd 
from sklearn import model_selection

import os 
os.chdir(r'/home/aditya/Kaggle/workspace/ml-proj-template')

if __name__ == '__main__':
    df = pd.read_csv(r'./input/train.csv')
    kf = model_selection.StratifiedKFold(n_splits=5,random_state=None, shuffle=False)
    df['kfold'] = -1

    for fold, (train_idx, test_idx) in enumerate(kf.split(X= df , y = df.target.values)):
        print(len(train_idx), len(test_idx))
        df.loc[test_idx,'kfold']=fold
    
    df.to_csv(r'./input/train_folds.csv',index= False)
    


