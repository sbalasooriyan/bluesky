'''
output.py

This script imports the processed data and outputs it.
'''

import numpy as np
import pickle
import pandas


pdFileName  = 'pd.pickle'

with open(pdFileName, 'rb') as handle:
    PD = pickle.load(handle)

colnames = np.unique(np.array(sorted(PD.method)))
rownames = np.array(["CFL","LOS","AC with MCFL","MCFL_max","CFL/AC"])
PD.numpify()
show_data = np.zeros((rownames.shape[0],colnames.shape[0]))
for i in range(show_data.shape[1]):
    ind = PD.method == colnames[i]
    N = float(sum(ind))
    show_data[0,i] = np.sum(PD.cfl_sum[ind])/N
    show_data[1,i] = np.sum(PD.los_sum[ind])/N
    show_data[2,i] = np.sum(PD.mcfl[ind])/N
    show_data[3,i] = np.sum(PD.mcfl_max[ind])/N
    show_data[4,i] = np.sum(PD.cfl_sum[ind]/PD.nac[ind])/N

pandas.set_option('expand_frame_repr', False)
print pandas.DataFrame(show_data,rownames,colnames)