'''
output.py

This script imports the processed data and outputs it.
'''

import numpy as np
import pickle
import pandas
import matplotlib.pyplot as plt

def replaceColNames(colnames):
    for i in range(len(colnames)):
        colnames[i] = colnames[i].replace('-OFF','')
        colnames[i] = colnames[i].replace('SSD-FF','SSD')
    return colnames

# Settings
pdFileName  = 'pd.pickle'
showTable   = True
showPlot    = True

with open(pdFileName, 'rb') as handle:
    PD = pickle.load(handle)


insts = np.unique(sorted(PD.inst))
PD.numpify()

if showTable:
    for inst in insts:
        idx = PD.inst == inst
        colnames = np.unique(np.array(sorted(PD.method[idx])))
        rownames = np.array(["CFL","LOS","AC with MCFL","MCFL_max","CFL/AC"])
        show_data = np.zeros((rownames.shape[0],colnames.shape[0]))
        for i in range(show_data.shape[1]):
            ind = np.logical_and(PD.method == colnames[i],idx)
            N = float(sum(ind))
            show_data[0,i] = np.sum(PD.cfl_sum[ind])/N
            show_data[1,i] = np.sum(PD.los_sum[ind])/N
            show_data[2,i] = np.sum(PD.mcfl[ind])/N
            show_data[3,i] = np.sum(PD.mcfl_max[ind])/N
            show_data[4,i] = np.sum(PD.cfl_sum[ind]/PD.nac[ind])/N
        
        pandas.set_option('expand_frame_repr', False)
        
        print '\n\033[4minst: ' + str(inst) + '    \033[0m'
        print pandas.DataFrame(show_data,rownames,colnames)

if showPlot:
    plt.close('all')
    datas = []
    fig, axes = plt.subplots(nrows=2, ncols=len(insts))
    k = -1
    for inst in insts:
        idx = PD.inst == inst
        colnames = np.unique(np.array(sorted(PD.method[idx])))
        # Remove OFF-OFF        
        colnames2 = np.unique(np.array(sorted(PD.method[idx])))
        colnames2 = np.delete(colnames2,np.where(colnames2 == 'OFF-OFF')[0])
        
        boxes = []
        boxes2 = []
        for col in colnames:
            ind = np.logical_and(PD.method == col,idx)
            boxes.append(PD.cfl_sum[ind]/PD.nac[ind])
        for col in colnames2:
            ind = np.logical_and(PD.method == col,idx)
            boxes2.append(PD.los_sum[ind]/PD.nac[ind])
        # Density
        dens = round(inst * 10000. / 455625.,1)
        k += 1
        axes[0,k].boxplot(boxes, labels=replaceColNames(colnames))
        axes[0,k].set_title('CFL/AC with ' + r'$\rho =$' + str(dens))
        axes[1,k].boxplot(boxes2, labels=replaceColNames(colnames2))
        axes[1,k].set_title('LOS/AC with ' + r'$\rho =$' + str(dens))
#        fig = plt.figure()
#        plt.boxplot(boxes)
#    for i in range(len(insts)):
    plt.show()