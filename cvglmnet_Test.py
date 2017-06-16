# -*- coding: utf-8 -*-

import os, sys
import pandas as pd, numpy as np
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings

import glmnet_python
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot
from cvglmnetPredict import cvglmnetPredict
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

def load_hdf(loc, df_name):
    # read hdf format data
    store = pd.HDFStore(loc)
    return(store[df_name])
    
def generateFoldid(xOrig, yOrig, nFold):
    kFold = StratifiedKFold(n_splits = nFold)
    foldId = np.empty(yOrig.shape, dtype = int)
    
    id = 0
    for trainIndex, testIndex in kFold.split(xOrig, yOrig):
        foldId[testIndex] = id
        id += 1
    return foldId

def computeLogLoss(cvFit, yOrig):
    ''' compute logLoss from cvglmnet fit
        cvFit: a cvglmnet fit object
        yOrig: real Y values
    '''
    logLossDict, minIdx = {}, np.argmin(fit['cvm'])
    optimalPred = fit['fit_preval'][:,minIdx]
    nfolds = scipy.amax(fit['foldid']) + 1
    
    for i in range(nfolds):
        which = (fit['foldid'] == i)
        yPred, yReal = optimalPred[which], yOrig[which]
        # compute logLoss using Sklearn package
        logLossDict[i] = log_loss(yReal, yPred)
    return logLossDict

def main(toPrint=True):
    os.chdir('/exeh/exe3/zhaok/data')
    loc = 'GSE92742_Broad_LINCS_Level5_COMPZ.N05A_n473647x12328.h5'
    df_name = 'gene_expr'
    nfold = 3
    pheno = load_hdf(loc, df_name)
    xOrig = pheno.iloc[:, 1:].values
    xOrig = xOrig.astype(np.float64)
    yOrig = pheno.iloc[:, 0].values
    yOrig = yOrig.astype(np.float64)
    
    drugName = np.asarray(pheno.index, dtype=str)    
    xOrig, yOrig, drugName = shuffle(xOrig, yOrig, drugName, random_state = 0)
    fid = generateFoldid(xOrig, yOrig, nfold)
    
    optimalAlpha = 0
    minLoss = sys.maxint
    lossDict = {}
    
    warnings.filterwarnings('ignore')    
    for alpha in np.arange(0, 1.1, 0.1): 
        fit = cvglmnet(x = xOrig, y = yOrig, alpha = alpha, family = 'binomial', ptype='deviance', foldid = fid, keep = True)
        logLossDict = computeLogLoss(fit, yOrig)
        avgLogLoss = np.mean(logLossDict.values())
        
        if minLoss > avgLogLoss:
            minLoss, lossDict, optimalAlpha = avgLogLoss, lossDict, alpha
        
        if toPrint == True:
            print('Alpha value: ', alpha, '; log loss for each fold: ', logLossDict.values())
    
    warnings.filterwarnings('default')
    
    print('optimalAlpha:', optimalAlpha)
    print('min average log loss:', minLoss)
    print('log loss for each cross validation: ', lossDict.values())
    
if __name__ == '__main__':
    main()
    
