'''
output.py

This script imports the processed data and outputs it.
'''

import numpy as np
import pickle
import pandas
import matplotlib.pyplot as plt
from latexify import latexify

def replaceColNames(colnames):
    for i in range(len(colnames)):
        colnames[i] = colnames[i].replace('-OFF','')
        colnames[i] = colnames[i].replace('SSD-FF','SSD')
    return colnames

# Settings
pdFileName  = 'check.pickle'
showTable   = True
showPlot    = True
showLatex   = False

with open(pdFileName, 'rb') as handle:
    PD = pickle.load(handle)


insts = np.unique(sorted(PD.inst))
# Replace nans with 0
PD.lostime[PD.lostime != PD.lostime] = 0.0
#PD.numpify()

if showTable:
    for inst in insts:
        idx = PD.inst == inst
        colnames = np.unique(np.array(sorted(PD.method[idx])))
        rownames = np.array(["CFL","LOS","LOS_INTIME","AC with MCFL","MCFL_max","CFL/AC",\
                             "TIME/AC","DIST/AC","WORK/AC","TIMEINCFL/CFL","TIMEINLOS/LOS","TIMEINCFL/AC","TIMEINLOS/AC","DENSITY"])
        show_data = np.zeros((rownames.shape[0],colnames.shape[0]))
        for i in range(show_data.shape[1]):
            ind = np.logical_and(PD.method == colnames[i],idx)
            N = float(sum(ind))
            show_data[0,i] = np.sum(PD.cfl_sum[ind])/N
            show_data[1,i] = np.sum(PD.los_sum[ind])/N
            show_data[2,i] = np.sum(PD.los_count[ind])/N
            show_data[3,i] = np.sum(PD.mcfl[ind])/N
            show_data[4,i] = np.sum(PD.mcfl_max[ind])/N
            show_data[5,i] = np.sum(PD.cfl_sum[ind]/PD.nac[ind])/N
            show_data[6,i] = np.sum(PD.time[ind])/N
            show_data[7,i] = np.sum(PD.dist[ind])/N
            show_data[8,i] = np.sum(PD.work[ind])/N
            show_data[9,i] = np.sum(PD.cfltime[ind])/N
            show_data[10,i] = np.sum(PD.lostime[ind])/N
            show_data[11,i] = np.sum(PD.cfltime[ind]*PD.cfl_sum[ind]/PD.nac[ind])/N
            show_data[12,i] = np.sum(PD.lostime[ind]*PD.los_sum[ind]/PD.nac[ind])/N
            show_data[13,i] = np.sum(PD.density[ind])/N
        
        pandas.set_option('expand_frame_repr', False)
        
        print '\n\033[4minst: ' + str(inst) + '    \033[0m'
        print pandas.DataFrame(show_data,rownames,colnames)

if showPlot:
#    plt.close('all')
#    datas = []
    fig, axes = plt.subplots(nrows=5, ncols=len(insts))
    fig2, axes2 = plt.subplots(nrows=5, ncols=len(insts))
    fig.canvas.set_window_title(pdFileName[:-7] + ' - 1')
    fig2.canvas.set_window_title(pdFileName[:-7] + ' - 2')
    k = -1
    for inst in insts:
        idx = PD.inst == inst
        colnames = np.unique(np.array(sorted(PD.method[idx])))
        # Remove OFF-OFF        
        colnames2 = np.unique(np.array(sorted(PD.method[idx])))
        colnames2 = np.delete(colnames2,np.where(colnames2 == 'OFF-OFF')[0])
        
        boxes = []
        boxes2 = []
        boxes3 = []
        boxes4 = []
        boxes5 = []
        boxes6 = []
        boxes7 = []
        boxes8 = []
        boxes9 = []
        boxes10 = []
        for col in colnames:
            ind = np.logical_and(PD.method == col,idx)
            boxes.append(PD.cfl_sum[ind]/PD.nac[ind])
            boxes3.append(PD.time[ind])
            boxes4.append(PD.dist[ind])
            boxes5.append(PD.work[ind])
            boxes9.append(PD.density[ind])
        for col in colnames2:
            ind = np.logical_and(PD.method == col,idx)
            boxes2.append(PD.los_sum[ind]/PD.nac[ind])
            boxes6.append(PD.cfltime[ind])
            boxes7.append(PD.lostime[ind])
            boxes8.append(PD.mcfl[ind])
            boxes10.append(PD.los_count[ind]/PD.nac[ind])
        # Density
        dens = round(inst * 10000. / 455625.,1)
        k += 1
        if len(insts) > 1:
            axes[0,k].boxplot(boxes, labels=replaceColNames(colnames))
            axes[0,k].set_title('CFL/AC with ' + r'$\rho =$' + str(dens))
            axes[1,k].boxplot(boxes2, labels=replaceColNames(colnames2))
            axes[1,k].set_title('LOS/AC with ' + r'$\rho =$' + str(dens))
            axes[2,k].boxplot(boxes8, labels=replaceColNames(colnames2))
            axes[2,k].set_title('AC in MCFL with ' + r'$\rho =$' + str(dens))
            axes[3,k].boxplot(boxes9, labels=replaceColNames(colnames))
            axes[3,k].set_title('Density with ' + r'$\rho =$' + str(dens))
            axes[4,k].boxplot(boxes10, labels=replaceColNames(colnames2))
            axes[4,k].set_title('LOS/AC counted with ' + r'$\rho =$' + str(dens))
            axes2[0,k].boxplot(boxes3, labels=replaceColNames(colnames))
            axes2[0,k].set_title('Travel time with ' + r'$\rho =$' + str(dens))
            axes2[1,k].boxplot(boxes4, labels=replaceColNames(colnames))
            axes2[1,k].set_title('Travel dist with' + r'$\rho =$' + str(dens))
            axes2[2,k].boxplot(boxes5, labels=replaceColNames(colnames))
            axes2[2,k].set_title('Work done with ' + r'$\rho =$' + str(dens))
            axes2[3,k].boxplot(boxes6, labels=replaceColNames(colnames2))
            axes2[3,k].set_title('Time in CFL with ' + r'$\rho =$' + str(dens))
            axes2[4,k].boxplot(boxes7, labels=replaceColNames(colnames2))
            axes2[4,k].set_title('Time in LOS with ' + r'$\rho =$' + str(dens))
        else:
            axes[0].boxplot(boxes, labels=replaceColNames(colnames))
            axes[0].set_title('CFL/AC with ' + r'$\rho =$' + str(dens))
            axes[1].boxplot(boxes2, labels=replaceColNames(colnames2))
            axes[1].set_title('LOS/AC with ' + r'$\rho =$' + str(dens))
            axes[2].boxplot(boxes8, labels=replaceColNames(colnames2))
            axes[2].set_title('AC in MCFL with ' + r'$\rho =$' + str(dens))
            axes[3].boxplot(boxes9, labels=replaceColNames(colnames))
            axes[3].set_title('Density with ' + r'$\rho =$' + str(dens))
            axes[4].boxplot(boxes10, labels=replaceColNames(colnames2))
            axes[4].set_title('LOS/AC counted with ' + r'$\rho =$' + str(dens))
            axes2[0].boxplot(boxes3, labels=replaceColNames(colnames))
            axes2[0].set_title('Travel time with ' + r'$\rho =$' + str(dens))
            axes2[1].boxplot(boxes4, labels=replaceColNames(colnames))
            axes2[1].set_title('Travel dist with' + r'$\rho =$' + str(dens))
            axes2[2].boxplot(boxes5, labels=replaceColNames(colnames))
            axes2[2].set_title('Work done with ' + r'$\rho =$' + str(dens))
            axes2[3].boxplot(boxes6, labels=replaceColNames(colnames2))
            axes2[3].set_title('Time in CFL with ' + r'$\rho =$' + str(dens))
            axes2[4].boxplot(boxes7, labels=replaceColNames(colnames2))
            axes2[4].set_title('Time in LOS with ' + r'$\rho =$' + str(dens))
#        fig = plt.figure()
#        plt.boxplot(boxes)
#    for i in range(len(insts)):
#    latexify()
    plt.show()

if showLatex:
    fig1 = plt.figure()