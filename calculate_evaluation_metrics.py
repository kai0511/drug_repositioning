import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

def show_plot(x, y, auc):
    plt.clf()
    plt.plot(x, y, lw = 2, color='navy',label='AUC curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('AUC={0:0.2f}'.format(auc))
    plt.legend(loc="lower left")
    plt.show()
    
if __name__ == '__main__':
    os.chdir('/exeh/exe3/zhaok/')
    directories = ["depressionANDanxiety", 'antidepression', 'antipsycho', 'scz']
    models = ['svm', 'bt', 'rf']
    
    data_location_dic = {
        'scz':"/exeh/exe3/sohc/cmap/SCZ_indication_orig_drug_expr_with_DrugNames_standardized.csv",
        'antipsycho':"data/Cmap_differential_expression_antipsycho.csv",
        'antidepression':"data/Cmap_differential_expression_antidepression.csv",
        'depressionANDanxiety':"data/Cmap_differential_expression_anxiety_depression.csv"
    }

    for m in models:
        for d in directories:
            pheno = pd.read_csv(data_location_dic[d])    
            X_orig = pheno.iloc[:, 2:].values 
            y_orig = pheno.iloc[:, 1].values
            
            # shuffle and define stratified KFold
            X_orig, y_orig = shuffle(X_orig, y_orig, random_state = 0)
            testFold = StratifiedKFold(n_splits = 3)
            
            pred_res = pd.read_csv('GE_result/%s/%s_result.out' % (d, m))
            y_pred = pred_res.iloc[:, 1].values
            y_true = pred_res.iloc[:, 2].values # code for elastic net
            
            spliting_list = []
            for train_index, test_index in testFold.split(X_orig, y_orig):
                y_test = y[test_index]
                spliting_list.append(y_test.shape[0])
            
            splited_y_pred = np.split(y_pred, [spliting_list[0], sum(spliting_list[2])])
            splited_y_true = np.split(y_true, [spliting_list[0], sum(spliting_list[2])])
            
            for i in range(3):
                fpr, tpr, thresholds = roc_curve(splited_y_true[i], splited_y_pred[i], pos_label = 1)  # AUC for ROC curve
                roc_auc = auc(fpr, tpr)

                precision, recall, thresholds = precision_recall_curve(splited_y_true[i], splited_y_pred[i], pos_label=1)   # AUC for Precision and Recall curve
                prc_auc = average_precision_score(y_true, y_pred, average="micro")
            
                print('%s, %s, %s, %f, %f' % (d, m, i, roc_auc, prc_auc))