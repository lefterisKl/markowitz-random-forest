import os
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)


names = list(set([x.split("_")[0] for x in os.listdir("results_binary_classification")]))

all_tt = []
for name in names:
    x = pd.read_csv(f"results_binary_classification/{name}_run.csv")
    x['dataset'] = name
    all_tt.append(x)


all_tt = pd.concat(all_tt,axis=0)
all_tt['method'] = all_tt['method'].map({'xgb':"XGB",'xt':'ERT','rf':"RF",'ivf_rf':'IVRF',"owf_rf":"OWRF", 'swf_rf':'SWRF','mrf_rf':'MRF' })

#Performance results sorted by APS, which computes PRAUC

aps_sorted = (all_tt.groupby(['dataset','method'])['aps','roc','f1_score','precision','recall','accuracy','elapsed'].mean().
 reset_index().sort_values(['dataset','aps'],ascending=(True,False)))
num_cols = aps_sorted.columns.values[aps_sorted.dtypes == "float64"].tolist()
aps_sorted[num_cols] = aps_sorted[num_cols].round(4)
print(aps_sorted.to_latex())

#Method Rankings

methods = aps_sorted.method.drop_duplicates().sort_values().tolist()
aps_sorted['rank'] = list(range(1,len(methods)+1))*len(names)
ranks = aps_sorted.groupby('method')['rank'].agg(['mean','min','max','std']).sort_values('mean')
ranks = np.transpose(ranks)
num_cols = ranks.columns.values[ranks.dtypes == "float64"].tolist()
ranks[num_cols] = ranks[num_cols].round(4)
print(ranks.to_latex())

#Friedman and Nemenyi tests
methods  = all_tt['method'].drop_duplicates().sort_values().tolist()
all_tt = all_tt.sort_values(['dataset','method'])
fdcs = []
nmns = []
names = sorted(names)
for name in names:
    from scipy.stats import friedmanchisquare
    fdcs.append(friedmanchisquare(*[ all_tt[(all_tt.dataset == name) & (all_tt.method==m)].aps.values for m in methods] ) )
    import scikit_posthocs as sp
    import pandas as pd
    data = np.transpose([ all_tt[(all_tt.dataset == name) & (all_tt.method==m)].aps.values for m in methods])
    # Conduct the Nemenyi test
    nmn = pd.DataFrame(sp.posthoc_nemenyi_friedman(data))
    nmn.index = methods
    nmn.columns = methods
    nmn.reset_index(inplace=True)
    nmn['dataset'] = name
    nmns.append(nmn)


NMN = pd.concat(nmns)
NMN = NMN[['dataset','index']+methods]
num_cols = NMN.columns.values[NMN.dtypes == "float64"].tolist()
NMN[num_cols] = NMN[num_cols].round(4).astype(str)
print(NMN.to_latex())