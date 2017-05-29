import os
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from classification_models import rbf_svm


def load_hdf(loc, df_name):
    # read hdf format data
    store = pd.HDFStore(loc)
    return(store[df_name])

def main():
    loc = 'GSE92742_Broad_LINCS_Level5_COMPZ.N05A_n473647x12328.h5'
    df_name = 'gene_expr'
    SVM_loss = []
    
    os.chdir('/exeh/exe3/zhaok/data')
    pheno = load_hdf(loc, df_name)
    
    #.values would transfer dataframe to an array; iloc is a function to index row/columns in dataframe
    X_orig = pheno.iloc[:, 1:].values 
    y_orig = pheno.iloc[:, 0].values
        
    X_orig, y_orig = shuffle(X_orig, y_orig, random_state = 0)
    
    testFold = StratifiedKFold(n_splits = 3)
    trainFold = StratifiedKFold(n_splits = 3)
    
    for train_index, test_index in testFold.split(X_orig, y_orig):
        X_train, y_train = X_orig[train_index], y_orig[train_index]
        X_test, y_test = X_orig[test_index], y_orig[test_index]
    
        SVM = rbf_svm(trainFold)
        
        SVM_loss.append(SVM.fit(X_train, y_train).score(X_test, y_test))
        # svm_result = SVM.fit(X_train, y_train).predict_proba(X_test)
        
    print('The neg log loss for SVM: %s, average loss: %s' % (SVM_loss, np.mean(SVM_loss)))

if __name__ == '__main__':   
    main()