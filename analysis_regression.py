import os
import pandas as pd
import numpy as np


names = list(set([x.split("_")[0] for x in os.listdir("results_regression")]))

all_tt = []
for name in names:
    x = pd.read_csv(f"results_regression/{name}_run.csv")
    x['dataset'] = name
    all_tt.append(x)

all_tt = pd.concat(all_tt,axis=0)

all_tt['method'] = all_tt['method'].map({'xgb':"XGB",'xt':'ERT','rf':"RF",'ivf_rf':'IVRF',"owf_rf":"OWRF", 'swf_rf':'SWRF','mrf_rf':'MRF' })
all_tt.rename(columns={"time":"elapsed"},inplace=True)

mae_sorted = (all_tt.groupby(['dataset','method'])['mae','rmse','mape','r2','elapsed'].mean().
 reset_index().sort_values(['dataset','mae'],ascending=(True,True)))
methods = mae_sorted['method'].drop_duplicates().tolist()
mae_sorted.groupby("method")['mae'].median()
mae_sorted.groupby("method")['mape'].median()
mae_sorted.groupby("method")['rmse'].mean()
mae_sorted['rank'] = list(range(1,len(methods)+1))*len(names)
mae_sorted.groupby('method')['rank'].agg(['mean','std']).sort_values('mean')


ranks = mae_sorted.groupby('method')['rank'].agg(['mean','min','max','std']).sort_values('mean')
import numpy as np
ranks = np.transpose(ranks)
num_cols = ranks.columns.values[ranks.dtypes == "float64"].tolist()
ranks[num_cols] = ranks[num_cols].round(4)
print(ranks.to_latex())



num_cols = mae_sorted.columns.values[mae_sorted.dtypes == "float64"].tolist()
mae_sorted[num_cols] = mae_sorted[num_cols].round(4)
mae_sorted.drop(columns=['rank','sortkey'],inplace=True)
print(mae_sorted.to_latex())


#Friedman and Nemenyi tests
methods  = all_tt['method'].drop_duplicates().sort_values().tolist()
fdcs = []
nmns = []
for name in names:
    from scipy.stats import friedmanchisquare
    fdcs.append(friedmanchisquare(*[ all_tt[(all_tt.dataset == name) & (all_tt.method==m)].mae.values for m in methods] ) )
    import scikit_posthocs as sp
    import pandas as pd
    data = np.transpose([ all_tt[(all_tt.dataset == name) & (all_tt.method==m)].mae.values for m in methods])
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