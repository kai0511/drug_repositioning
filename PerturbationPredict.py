import os
import re
import pandas as pd
import numpy as np
from ClassificationModels import SVM

def getParams():
    '''return setting parameters for SVM'''
    return {'C': [1, 10, 100, 1000],  
            'gamma': [0.001, 0.0001], 
            'kernel': ['rbf'],
            'metrics': 'neg_log_loss',
            'num_jobs': 3}

def getSearchPattern(loc, name):
    '''
    loc the location of file, containing only one column for indicated drug list
    name a string specifies the name of the column
    return: a string concatenated by '|'
    '''
    drugList = pd.read_csv(loc, header=None)
    drugList.columns = [name]
    drugList = [d.lower() for d in drugList[name]]
    sePattern = "|".join(drugList)
    return sePattern

def getIndication(sePattern, drugList):
    ''' 
    use re package to find whether elements in drugList matches sePattern
    return: a list with 1 indicating the matching the pattern, otherwise not
    '''
    idx = map(lambda x: int(bool(re.search(sePattern, x, re.IGNORECASE))), drugList)
    return idx
    
def main():
    
    os.chdir('/exeh/exe3/zhaok/GE_result/genePerturbation')
    
    loc = '/exeh/exe3/zhaok/data/N05A.txt'
    sePattern = getSearchPattern(loc, 'drugName')
    
    drugBank = pd.read_table('consensi-drugbank.tsv', header=0)
    drugVoc = pd.read_csv('/exeh/exe3/zhaok/data/drugbank vocabulary.csv', header=0).iloc[:,[0,2]]
    
    # merge two data frames
    drugVoc.columns = ['DBID', 'drugName']
    newDrugBank = pd.merge(drugVoc, drugBank, left_on = 'DBID', right_on = 'perturbagen', suffixes=('', ''))
    newDrugBank = newDrugBank.iloc[:, 1:]
    
    # generate indication
    drugList = newDrugBank.loc[:,'drugName']
    indication = getIndication(sePattern, drugList)
    
    # .values would transfer dataframe to an array
    # .iloc is a function to index row/columns in dataframe
    X_orig = newDrugBank.iloc[:, 2:].values
    y_orig = np.asarray(indication)

    X_orig, y_orig = shuffle(X_orig, y_orig, random_state = 0)
    
    # fit model and yield cross validation result
    params = getParams()
    trainFold = StratifiedKFold(n_splits = 3)
    SVMObj = SVM(params, trainFold)
    SVMObj.fit(X_orig, y_orig)
    print(SVMObj.cv_results_)
    
    predictionData = {'knockdown': 'consensi-knockdown.tsv',
                      'overexpression':'consensi-overexpression.tsv',
                      'pert_id':'consensi-pert_id.tsv'}
    
    # make predictions on knockdonw, overexpression and all pert datasets
    for pertType in predictionData.keys():
        pert = pd.read_table(predictionData[pertType], header=0)
        pertRes = SVMObj.predict_proba(pert.iloc[:, 1:].values)
        resDict = {"predRes":pd.Series(pertRes), 'pertId': pert.loc[:, 'perturbagen']}
        resDataFrame = pd.DataFrame(resDict)
        resDataFrame.to_csv('SVM-'+pertType+'Res.csv')

if __name__ == '__main__':
    main()
