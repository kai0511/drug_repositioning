import os
import re
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress import parse


if __name__ == '__main__':

os.chdir('/exeh/exe3/zhaok/data')

# extract all samples that were treated with vorinostat
sig_info = pd.read_csv("GSE92742_Broad_LINCS_sig_info.txt", sep="\t", dtype=str)

# unique the pert name set
pert_set = set(sig_info["pert_iname"])

# read N05A data file
drug_list = pd.read_csv('N05A.txt', header=None)
drug_list.columns = ['names']
drug_list = [d.lower() for d in drug_list['names']]

# search pattern
se_pattern = "|".join(drug_list) 

# find the fuzzy intersection set
# search_res = [s for s in pert_set if bool(re.search(se_pattern, s)) == True]
search_res = filter(lambda x: bool(re.search(se_pattern, x)), pert_set)

idx = np.repeat(False, sig_info.shape[0])

for s in search_res:
    # print(sum(sig_info["pert_iname"] == s))
    idx[sig_info["pert_iname"] == s] = True

# generate positive observations    
pos_sig_id = sig_info["sig_id"][idx]

# print its size
print("number of samples treated with N051A drug: %s" % len(pos_sig_id))

# parse gene expression data using pos_sig_id obtained in the above step 
positive_gctx = parse('GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx', cid=pos_sig_id)

# (positive_gctx.data_df.T).to_csv('GSE92742_Broad_LINCS_Level5_COMPZ_N05A.csv')

# generate negative observations    
neg_sig_id = sig_info["sig_id"][~idx]

negative_gctx = parse('GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx', cid=neg_sig_id)

# append positive gctx to negative gctx
gene_expr = (negative_gctx.data_df.T).append(positive_gctx.data_df.T)

# write them to a csv file
gene_expr.to_csv('GSE92742_Broad_LINCS_Level5_COMPZ_N05A.csv')
