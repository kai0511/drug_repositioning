import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold 

if __name__ == '__main__':
    os.chdir("/exeh/exe3/zhaok/GE_result")
    #pheno = pd.read_csv("/exeh/exe3/sohc/cmap/SCZ_indication_orig_drug_expr_with_DrugNames_standardized.csv")
    pheno = pd.read_csv("/exeh/exe3/zhaok/data/Cmap_differential_expression_antipsycho.csv")
    y_orig = pheno.iloc[:, 1].values

    pheno, y = shuffle(pheno.as_matrix(), y_orig, random_state=0)
    threeFold = StratifiedKFold(n_splits = 3)
    
    num = 1    
    for train_index, test_index in threeFold.split(pheno, y):
        pheno_train, pheno_test = pheno[train_index], pheno[test_index]
        file_train = 'Cmap_differential_expression_antipsycho_train_part%s.csv' % num
        file_test = 'Cmap_differential_expression_antipsycho_test_part%s.csv' % num
        pd.DataFrame(pheno_train).to_csv(file_train, header=False, index=False)
        pd.DataFrame(pheno_test).to_csv(file_test, header=False, index=False)
        num+=1
