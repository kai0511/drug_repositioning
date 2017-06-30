from __future__ import print_function

import os
import re
import warnings
import hyperas
import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adam, RMSprop, Adagrad, Adamax
from keras.regularizers import l1,l2, l1l2, activity_l2, activity_l1, activity_l1l2

# warnings.filterwarnings('ignore', message='Changing the shape of non-C contiguous array')  # filter out specific warning
warnings.filterwarnings("ignore")

class DeepNeuronNetwork(object):

    def __init__(self, X, y, fileName, maxEvals = 50, numFold = 3, standardize=False):
        ''' X input data, type: numpy.ndarray 
            y expected output, type: numpy.ndarray 
            params python dictionary used to set up neuron network
        '''
        # before fitting the model, remember to standardize the input! (as we do in our training)
        # .values would transfer dataframe to an array; 
        # .iloc is a function to index row/columns in dataframe
        if standardize:
            self.inputData = StandardScaler().fit_transform(X)  
        else:
            self.inputData = X
            
        self.outputData = y
        self.numFeature = self.inputData.shape[1]
        self.train_X, self.train_y = None, None
        self.test_X, self.test_y = None, None
        self.model = None
        self.numFold = numFold
        self.params = {}
        self.minLosses = []
        self.models = []
        self.maxEvals = maxEvals
        self.fileName = fileName  #'keras_deep_learning_prediction.out'

    def setParams(self):
        '''
        # KEY: for unbalanced dataset, be careful with the choice of minibatch size in each epoch.
        If you choose too few, eg 50, then all validation sample in the minibatch may have the same label 
        (validation loss will be the the same if all outcome variable is 0) 
        '''
        params = {'choice': hp.choice('num_layers',
                                     [{'layers':'two', },
                                      {'layers':'three', 
                                       'units3': hp.uniform('units3', 64,1024), 
                                       'dropout3': hp.uniform('dropout3', .25,.75)}]),
                 
                 'units1': hp.uniform('units1', 64,1024),
                 'units2': hp.uniform('units2', 64,1024),
                 
                 'dropout1': hp.uniform('dropout1', .25,.75),
                 'dropout2': hp.uniform('dropout2',  .25,.75),
                 
                 # 'batch_size' : hp.uniform('batch_size', 256,1024),
                 'batch_size' : self.train_y.shape[0], # key change for unbalanced dataset 
                 'l1_1st_layer': hp.uniform('l1_1st_layer',1e-5,1e-2), #switch order of the two parameters
                 'l2_1st_layer': hp.uniform('l2_1st_layer',1e-5,1e-2),
                 'nb_epochs' : 100, 
                 'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
                 'activation':  hp.choice('activation',['relu','softplus','tanh'])}
                 
        self.params = params
        
    def constructNeuronNetwork(self, params):
        ''' construct neuron network using Keras Sequential
        '''
        model = Sequential()
        model.add(Dense(input_dim = self.numFeature, 
                        output_dim = params['units1'], 
                        W_regularizer = l1l2(l1=params['l1_1st_layer'],l2=params['l2_1st_layer'])))
                        
        # model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        model.add(Dropout(params['dropout1']))
        
        model.add(Dense(output_dim=params['units2'], 
                        W_regularizer=l1l2(l1=params['l1_1st_layer'],l2=params['l2_1st_layer']), 
                        init = "glorot_uniform")) 
        # model.add(BatchNormalization())
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))
        
        if params['choice']['layers'] == 'three':
            model.add(Dense(output_dim=params['choice']['units3'], 
                            W_regularizer=l1l2(l1=params['l1_1st_layer'],l2=params['l2_1st_layer']), 
                            init = "glorot_uniform"))
            model.add(BatchNormalization())
            model.add(Activation(params['activation']))
            model.add(Dropout(params['choice']['dropout3']))    
        
        model.add(Dense(1))
        # model.add(BatchNormalization())
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])
        
        self.model = model

    def fit(self, params):
        # save the the best weights out of all epoch runs
        self.constructNeuronNetwork(params)
        callback1 = ModelCheckpoint(filepath = "callback_prediction.hdf5", 
                                    verbose = 0, 
                                    save_best_only = True)
        self.model.fit(self.train_X, 
                       self.train_y,  
                       batch_size = params['batch_size'], 
                       nb_epoch = params['nb_epochs'], 
                       validation_data = (self.test_X, self.test_y), 
                       callbacks = [callback1],
                       show_accuracy = True,
                       verbose = 0)
                  
        model_json = self.model.to_json()
        del(self.model)
            
        self.model = model_from_json(model_json)
        self.model.load_weights("callback_prediction.hdf5")
        
        # latest version of keras requires re-compilation of model
        self.model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])  
        
        training_loss = self.eval(self.train_X, self.train_y)
        print('loss on the training set: %f.' % training_loss)

        validation_loss = self.eval(self.test_X, self.test_y)
        print('loss on the validation set: %f.' % validation_loss)

        return {'loss': validation_loss, 'status': STATUS_OK, 'model': self.model}        

    def eval(self, evaluate_X, evaluate_y):
        loss = self.model.evaluate(evaluate_X, 
                                   evaluate_y, 
                                   batch_size = evaluate_y.shape[0], 
                                   verbose = 0, 
                                   sample_weight=None)
        return loss
        
    def predict(self, X):        
        predRes = self.model.predict(X, batch_size = X.shape[0], verbose = 0) 
        return predRes[:,0]
    
    def runCrossValidation(self):
        os.system('touch -f %s && > %s' % (self.fileName, self.fileName))
        neg_log_loss = []
        
        # create three arrays to map corresponding parameters at the prediction stage  
        # you can use the following 2 lines for each new classifier, should return the same repeated stratified K-fold each time 
        # alternative: skf = StratifiedKFold(y, n_fold, shuffle=True, random_state=j*10)          
        # print cross_val_score(clf, X, y, cv=skf)
        self.inputData, self.outputData = shuffle(self.inputData, self.outputData, random_state=0)
        
        trainFold = StratifiedKFold(n_splits = self.numFold)
        
        for train_idx, test_idx in trainFold.split(self.inputData, self.outputData):
            self.train_X, self.test_X = self.inputData[train_idx], self.inputData[test_idx] 
            self.train_y, self.test_y = self.outputData[train_idx], self.outputData[test_idx]
            
            self.setParams()
            trials = Trials()
            optimalParam = fmin(self.fit, self.params, algo=tpe.suggest, max_evals = self.maxEvals, trials=trials) 
            
            # stores the min loss and the best model of each CV run
            # ref: https://districtdatalabs.silvrback.com/parameter-tuning-with-hyperopt
            lossRes = [res['loss'] for res in trials.results if res['status'] == 'ok']
            modelRes = [res['model'] for res in trials.results if res['status'] == 'ok']
            minPos = np.argmin(lossRes)
            self.minLosses.append(lossRes[minPos])
            self.models.append(modelRes[minPos])
                        
            print('Best parameters estimates: \n%s\n' % optimalParam) 
            print('Corresponding loss using the best combination of parameters: \n%s\n' % lossRes[minPos])
            
        idx = np.argmin(self.minLosses)
        print('The index for choosing the best parameter: %s' % idx) 
        print('The min loss list for cross validation: \n%s\n' % lossRes)
        self.model = self.models[idx]
        print('The json formate best parameters: \n%s\n' % self.model.to_json())

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
    os.chdir("/exeh/exe3/zhaok/GE_result")
    fileName = 'genePerturbation/keras_deep_learning_prediction.out' 
    
    loc = '/exeh/exe3/zhaok/data/N05A.txt'
    sePattern = getSearchPattern(loc, 'drugName')
    
    drugBank = pd.read_table("genePerturbation/consensi-drugbank.tsv", header=0)
    drugVoc = pd.read_csv('/exeh/exe3/zhaok/data/drugbank vocabulary.csv', header=0).iloc[:,[0,2]]
    
    # merge two data frames
    drugVoc.columns = ['DBID', 'drugName']
    newDrugBank = pd.merge(drugVoc, drugBank, left_on = 'DBID', right_on = 'perturbagen', suffixes=('', ''))
    newDrugBank = newDrugBank.iloc[:, 1:]
    
    # generate indication
    drugList = newDrugBank.loc[:,'drugName']
    indication = getIndication(sePattern, drugList)
    
    X_orig = newDrugBank.iloc[:, 2:].values
    y_orig = np.asarray(indication)
    
    DNN = DeepNeuronNetwork(X_orig, y_orig, fileName)
    DNN.runCrossValidation()
    print ("The negative log loss for Deep Learning: %s, average log loss: %s" % (DNN.minLosses, np.mean(DNN.minLosses)))
    
    predictionData = {'knockdown': 'consensi-knockdown.tsv',
                      'overexpression':'consensi-overexpression.tsv',
                      'pert_id':'consensi-pert_id.tsv'}
    
    for pertType in predictionData.keys():
        pert = pd.read_table(predictionData[pertType], header=0)
        pertRes = DNN.predict(pert.iloc[:, 1:].values)
        resDict = {"predRes": pd.Series(pertRes), 'pertId': pert.loc[:, 'perturbagen']}
        resDataFrame = pd.DataFrame(resDict)
        resDataFrame.to_csv('DNN-'+pertType+'Res.csv')

if __name__ == '__main__':    
    main()
