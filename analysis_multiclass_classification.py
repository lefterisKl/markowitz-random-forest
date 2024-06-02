import os
import pandas as pd
import numpy as np



names = list(set([x.split("_")[0] for x in os.listdir("results_multiclass_classification")]))

all_tt = []
for name in names:
    x = pd.read_csv(f"results_multiclass_classification/{name}_run.csv")
    x['dataset'] = name
    all_tt.append(x)

all_tt = pd.concat(all_tt,axis=0)
all_tt['method'] = all_tt['method'].map({'xgb':"XGB",'xt':'ERT','rf':"RF",'ivf_rf':'IVRF',"owf_rf":"OWRF", 'swf_rf':'SWRF','mrf_rf':'MRF' })
methods = all_tt['method'].drop_duplicates().tolist()

f1M_sorted = (all_tt.groupby(['dataset','method'])['f1_macro','f1_micro','accuracy','elapsed'].mean().
reset_index().sort_values(['dataset','f1_macro'],ascending=(True,False)))
f1M_sorted['rank'] = list(range(1,len(methods)+1))*len(names)
f1M_sorted.groupby('method')['rank'].agg(['mean','std']).sort_values('mean')
ranks = f1M_sorted.groupby('method')['rank'].agg(['mean','min','max','std']).sort_values('mean')
ranks = np.transpose(ranks)
num_cols = ranks.columns.values[ranks.dtypes == "float64"].tolist()
ranks[num_cols] = ranks[num_cols].round(4)
print(ranks.to_latex())
num_cols = f1M_sorted.columns.values[f1M_sorted.dtypes == "float64"].tolist()
f1M_sorted[num_cols] = f1M_sorted[num_cols].round(4)
f1M_sorted.drop(columns=[ 'rank'],inplace=True)
print(f1M_sorted.to_latex())



methods  = all_tt['method'].drop_duplicates().sort_values().tolist()
fdcs = []
nmns = []
for name in names:
    from scipy.stats import friedmanchisquare
    fdcs.append(friedmanchisquare(*[ all_tt[(all_tt.dataset == name) & (all_tt.method==m)].f1_macro.values for m in methods] ) )
    import scikit_posthocs as sp
    import pandas as pd
    # Assuming 'metric' is a column for your metric of interest
    data = np.transpose([ all_tt[(all_tt.dataset == name) & (all_tt.method==m)].f1_macro.values for m in methods])
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