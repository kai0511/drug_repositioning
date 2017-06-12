import pandas as pd
import os

def rankDataFrame(fileName, colName, sort_idx, toFile):
    coefs = pd.read_csv(fileName, header=None)
    coef_df = coefs.loc[:,1:]
    coef_df.columns = colName
    rankedCoefs = coef_df.sort_values(by = colName[sort_idx], ascending = False)
    rankedCoefs.to_csv(toFile)


if __name__ == '__main__':
    
    os.chdir('/exeh/exe3/zhaok/GE_result/coefs')
    colName = ['gene_id', 'p1', 'p2', 'p3', 'avg']
    diseases = ['antidepression', 'antipsycho', 'depressionANDanxiety', 'scz'] 
    
    for d in diseases:
        fileName = '%s_coef.csv' % d
        toFile = '%s_coef_ranked.csv' % d
        rankDataFrame(fileName, colName, 4, toFile)
