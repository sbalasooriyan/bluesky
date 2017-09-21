'''
processor2.py

This script processes the log-files. Returns an object containing all data
needed for plotting and analysis.
'''

import os
import numpy as np
import pickle
import pandas

import warnings
from dataclasses import ProcessedData, ACData, CFLLOS
warnings.simplefilter('always')


        
            
    
def printError(n, varA=None, varB=None, varC=None):
    if n == 0:
        print "%s was not created, cannot delete" %(varA)
    elif n == 1:
        print "Entry %s is starting CFL/LOS, but not coupled" %(varA)
    elif n == 2:
        print "Entry %s is ending CFL/LOS, but not coupled" %(varA)
    elif n == 3:
        print "Ending CFL/LOS %s is impossible, it does not exist" %(varA)
    elif n == 4:
        print "Ending CFL/LOS %s is impossible, it is already closed" %(varA)
    elif n == 5:
        print "Combi %s and its ids do not match" %(varA)
    elif n == 6:
        print "Starting CFL/LOS %s is impossible, it is already open" %(varA)


# Storage folders for log-files
logFilesDir = './final_10_12345/'
pdFileName  = logFilesDir[2:-1] + '.pickle'
Areas       = [455625.0, 810000.0]
countlos    = 10.

# ProcessedData
PD = ProcessedData()

#%% Step 1: Get the log files
logFiles = [f for f in os.listdir(logFilesDir) if f.count(".log")>0 and f.count("SKYLOG")>0]

for log in logFiles:
#for k in [0]:
#    log = logFiles[k]
    # Print progress
    print 'Processing (%d/%d): %s' %(logFiles.index(log)+1,len(logFiles),log)
    #log = logFiles[1]
    # Isolate scenario-name
    scn, _, _ = log.split(";")
    # Get method, vrange, inst and rep
    method, vrange, inst, rep = scn.split("_")
    # Get reso and rule
    reso, rule = method.split("-")
#    # Vrange in integers
#    vrange = map(int, vrange.split("-"))
    # Inst
    inst = int(inst[4:])
    # Repetition
    rep = int(rep[3:])
    
    # Read log-file and store in data
    data = pandas.read_csv(logFilesDir+log, comment='#', delimiter=',', header=None)
    data = data.as_matrix()
    entries = np.shape(data)[0]
#    print np.array([min(data[data[:,5]==10972.80000000,7]),max(data[:,7])])/1852*3600
#    print 1.*sum(np.logical_and(data[data[:,5]==10972.80000000,7]<=501*1852/3600,data[data[:,5]==10972.80000000,7]>=449*1852/3600))/sum(data[:,5]==10972.80000000)
#    # Aircraft
#    AC  = np.unique(data[:,1])
#    nAC = np.shape(AC)[0]
    
#    # For every entry, check what type it is
#    data_type = np.zeros(entries, dtype=np.int32)
#    conv = {"CRE": 1, "DEL": 2, "CFL": 3, "LOS": 4}
#    for i in range(entries):
#        data_type[i] = conv[data[i,2][:3]]
#    
##    # Perform sanity check
##    sanity(data, data_type, log, nAC)
#    
#    # Make array for flying AC
#    fly = np.zeros(nAC, dtype=bool)
#    
#    # Make current conflict-lists
#    cfl = [[] for x in xrange(nAC)]
#    cfl_hist = [[] for x in xrange(nAC)]
#    los = [[] for x in xrange(nAC)]
#    los_hist = [[] for x in xrange(nAC)]
#    
#    cfl_count = np.zeros(nAC, dtype=np.int32)
#    los_count = np.zeros(nAC, dtype=np.int32)
    
    PD.method.append(method)
    PD.inst.append(inst)
    PD.rep.append(rep)
    PD.vrange.append(vrange)
    
    ACD = ACData()
    CD  = CFLLOS(False)
    LD  = CFLLOS(True)
    skip = False
    for i in range(entries):
        # Sometimes the entry has been processed
        if skip:
            skip = False
            continue

        # Create AC
        if data[i,2][:3] == "CRE":
            ACD.cre(data[i,1],data[i,0])

        # Delete AC
        elif data[i,2][:3] == "DEL":
            try:
                idx = ACD.id.index(data[i,1])
                ACD.dlt(idx, data[i,0], data[i,17], data[i,18])
            except ValueError:
                printError(0, data[i,0])

        # CFL entry
        elif data[i,2][:3] == "CFL":
            [id0, id1] = sorted([data[i,1],data[i,2][10:]])
            combi = id0 + ' ' + id1
            if data[i,2][4:9] == "START":
                if not data[i+1,2][4:9] == "START":
                    printError(1, i)
                # MCFL
                ACD.cfl_cur[ACD.id.index(id0)] += 1
                ACD.cfl_cur[ACD.id.index(id1)] += 1
                ACD.mcfl[ACD.id.index(id0)] = max(ACD.cfl_cur[ACD.id.index(id0)], ACD.mcfl[ACD.id.index(id0)])
                ACD.mcfl[ACD.id.index(id1)] = max(ACD.cfl_cur[ACD.id.index(id1)], ACD.mcfl[ACD.id.index(id1)])

                if not combi in CD.combi:
                    # New combi
                    CD.new(combi, id0, id1, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                    skip = True
                    # Edit time/dist/work in cfl
                    if ACD.cfl_cur[ACD.id.index(id0)] == 1:
                        ACD.cfltime[ACD.id.index(id0)] = data[i,0]  - ACD.cfltime[ACD.id.index(id0)]
                        ACD.cfldist[ACD.id.index(id0)] = data[i,17] - ACD.cfldist[ACD.id.index(id0)]
                        ACD.cflwork[ACD.id.index(id0)] = data[i,18] - ACD.cflwork[ACD.id.index(id0)]
                    if ACD.cfl_cur[ACD.id.index(id1)] == 1:
                        ACD.cfltime[ACD.id.index(id1)] = data[i,0]    - ACD.cfltime[ACD.id.index(id1)]
                        ACD.cfldist[ACD.id.index(id1)] = data[i+1,17] - ACD.cfldist[ACD.id.index(id1)]
                        ACD.cflwork[ACD.id.index(id1)] = data[i+1,18] - ACD.cflwork[ACD.id.index(id1)]
                else:
                    # Reopen existing
                    idx = CD.combi.index(combi)
                    if id0 == data[i,1] and id1 == data[i,2][10:]:
                        order = True
                    elif id1 == data[i,1] and id0 == data[i,2][10:]:
                        order = False
                    else:
                        printError(5, combi)
                    if not CD.open[idx]:
                        CD.cnt(idx, order, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                        skip = True
                        # Edit time/dist/work in cfl
                        if ACD.cfl_cur[ACD.id.index(id0)] == 1:
                            ACD.cfltime[ACD.id.index(id0)] = data[i,0]  - ACD.cfltime[ACD.id.index(id0)]
                            ACD.cfldist[ACD.id.index(id0)] = data[i,17] - ACD.cfldist[ACD.id.index(id0)]
                            ACD.cflwork[ACD.id.index(id0)] = data[i,18] - ACD.cflwork[ACD.id.index(id0)]
                        if ACD.cfl_cur[ACD.id.index(id1)] == 1:
                            ACD.cfltime[ACD.id.index(id1)] = data[i,0]    - ACD.cfltime[ACD.id.index(id1)]
                            ACD.cfldist[ACD.id.index(id1)] = data[i+1,17] - ACD.cfldist[ACD.id.index(id1)]
                            ACD.cflwork[ACD.id.index(id1)] = data[i+1,18] - ACD.cflwork[ACD.id.index(id1)]
                    else:
                        printError(6, combi)
            elif data[i,2][4:9] == "ENDED":
                if not data[i+1,2][4:9] == "ENDED":
                    printError(2, i)
                [id0, id1] = sorted([data[i,1],data[i,2][10:]])
                combi = id0 + ' ' + id1
                # MCFL
                ACD.cfl_cur[ACD.id.index(id0)] -= 1
                ACD.cfl_cur[ACD.id.index(id1)] -= 1
                try:
                    idx = CD.combi.index(combi)
                    if id0 == data[i,1] and id1 == data[i,2][10:]:
                        order = True
                    elif id1 == data[i,1] and id0 == data[i,2][10:]:
                        order = False
                    else:
                        printError(5, combi)
                    if CD.open[idx]:
                        CD.end(idx, order, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                        skip = True
                        # Edit time/dist/work in cfl
                        if ACD.cfl_cur[ACD.id.index(id0)] == 0:
                            ACD.cfltime[ACD.id.index(id0)] = data[i,0]  - ACD.cfltime[ACD.id.index(id0)]
                            ACD.cfldist[ACD.id.index(id0)] = data[i,17] - ACD.cfldist[ACD.id.index(id0)]
                            ACD.cflwork[ACD.id.index(id0)] = data[i,18] - ACD.cflwork[ACD.id.index(id0)]
                        if ACD.cfl_cur[ACD.id.index(id1)] == 0:
                            ACD.cfltime[ACD.id.index(id1)] = data[i,0]    - ACD.cfltime[ACD.id.index(id1)]
                            ACD.cfldist[ACD.id.index(id1)] = data[i+1,17] - ACD.cfldist[ACD.id.index(id1)]
                            ACD.cflwork[ACD.id.index(id1)] = data[i+1,18] - ACD.cflwork[ACD.id.index(id1)]
                    else:
                        printError(4, combi)
                except ValueError:
                    printError(3, combi)

        # LOS entry
        elif data[i,2][:3] == "LOS":
            [id0, id1] = sorted([data[i,1],data[i,2][10:]])
            combi = id0 + ' ' + id1
            if data[i,2][4:9] == "START":
                if not data[i+1,2][4:9] == "START":
                    printError(1, i)
                # MCFL
                ACD.los_cur[ACD.id.index(id0)] += 1
                ACD.los_cur[ACD.id.index(id1)] += 1
                if not combi in LD.combi:
                    # New combi
                    LD.new(combi, id0, id1, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                    if data[i,0] - ACD.t0[ACD.id.index(id0)] <= countlos or data[i,0] - ACD.t0[ACD.id.index(id1)] <= countlos:
                        LD.count[-1] = False
                    skip = True
                    # Edit time/dist/work in los
                    if ACD.los_cur[ACD.id.index(id0)] == 1:
                        ACD.lostime[ACD.id.index(id0)] = data[i,0]  - ACD.lostime[ACD.id.index(id0)]
                        ACD.losdist[ACD.id.index(id0)] = data[i,17] - ACD.losdist[ACD.id.index(id0)]
                        ACD.loswork[ACD.id.index(id0)] = data[i,18] - ACD.loswork[ACD.id.index(id0)]
                    if ACD.los_cur[ACD.id.index(id1)] == 1:
                        ACD.lostime[ACD.id.index(id1)] = data[i,0]    - ACD.lostime[ACD.id.index(id1)]
                        ACD.losdist[ACD.id.index(id1)] = data[i+1,17] - ACD.losdist[ACD.id.index(id1)]
                        ACD.loswork[ACD.id.index(id1)] = data[i+1,18] - ACD.loswork[ACD.id.index(id1)]
                else:
                    # Reopen existing
                    idx = LD.combi.index(combi)
                    if id0 == data[i,1] and id1 == data[i,2][10:]:
                        order = True
                    elif id1 == data[i,1] and id0 == data[i,2][10:]:
                        order = False
                    else:
                        printError(5, combi)
                    if not LD.open[idx]:
                        LD.cnt(idx, order, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                        skip = True
                        # Edit time/dist/work in los
                        if ACD.los_cur[ACD.id.index(id0)] == 1:
                            ACD.lostime[ACD.id.index(id0)] = data[i,0]  - ACD.lostime[ACD.id.index(id0)]
                            ACD.losdist[ACD.id.index(id0)] = data[i,17] - ACD.losdist[ACD.id.index(id0)]
                            ACD.loswork[ACD.id.index(id0)] = data[i,18] - ACD.loswork[ACD.id.index(id0)]
                        if ACD.los_cur[ACD.id.index(id1)] == 1:
                            ACD.lostime[ACD.id.index(id1)] = data[i,0]    - ACD.lostime[ACD.id.index(id1)]
                            ACD.losdist[ACD.id.index(id1)] = data[i+1,17] - ACD.losdist[ACD.id.index(id1)]
                            ACD.loswork[ACD.id.index(id1)] = data[i+1,18] - ACD.loswork[ACD.id.index(id1)]
                    else:
                        printError(6, combi)
            elif data[i,2][4:9] == "ENDED":
                if not data[i+1,2][4:9] == "ENDED":
                    printError(2, i)
                [id0, id1] = sorted([data[i,1],data[i,2][10:]])
                combi = id0 + ' ' + id1
                # MCFL
                ACD.los_cur[ACD.id.index(id0)] -= 1
                ACD.los_cur[ACD.id.index(id1)] -= 1
                try:
                    idx = LD.combi.index(combi)
                    if id0 == data[i,1] and id1 == data[i,2][10:]:
                        order = True
                    elif id1 == data[i,1] and id0 == data[i,2][10:]:
                        order = False
                    else:
                        printError(5, combi)
                    if LD.open[idx]:
                        LD.end(idx, order, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18], data[i,20])
                        skip = True
                        # Edit time/dist/work in los
                        if ACD.los_cur[ACD.id.index(id0)] == 0:
                            ACD.lostime[ACD.id.index(id0)] = data[i,0]  - ACD.lostime[ACD.id.index(id0)]
                            ACD.losdist[ACD.id.index(id0)] = data[i,17] - ACD.losdist[ACD.id.index(id0)]
                            ACD.loswork[ACD.id.index(id0)] = data[i,18] - ACD.loswork[ACD.id.index(id0)]
                        if ACD.los_cur[ACD.id.index(id1)] == 0:
                            ACD.lostime[ACD.id.index(id1)] = data[i,0]    - ACD.lostime[ACD.id.index(id1)]
                            ACD.losdist[ACD.id.index(id1)] = data[i+1,17] - ACD.losdist[ACD.id.index(id1)]
                            ACD.loswork[ACD.id.index(id1)] = data[i+1,18] - ACD.loswork[ACD.id.index(id1)]
                    else:
                        printError(4, combi)
                except ValueError:
                    printError(3, combi)
    
    
    CD.numpify()
    LD.numpify()
    
#    for i in range(len(ACD.id)):
#        # Bools
#        idxC0 = ACD.id[i] == CD.id0
#        idxC1 = ACD.id[i] == CD.id1
#        idxC = np.where(np.logical_or(idxC0, idxC1))[0]
#        
#        idxL0 = np.in1d(LD.id0, ACD.id[i])
#        idxL1 = np.in1d(LD.id1, ACD.id[i])
#        idxL = np.where(np.logical_or(idxL0, idxL1))[0]
#        # Fill CFLs
#        ACD.cfl[i] = idxC
#        # Fill LOSs
#        ACD.los[i] = idxL
#        # Other stuff
#        if len(CD.combi) > 0:
#            ACD.cfltime[i] = np.sum(CD.time[idxC])
#            ACD.cfldist[i] = np.sum(CD.dist0[idxC0]) + np.sum(CD.dist1[idxC1])
#            ACD.cflwork[i] = np.sum(CD.work0[idxC0]) + np.sum(CD.work1[idxC1])
#        if len(LD.combi) > 0:
#            ACD.lostime[i] = np.sum(LD.time[idxL])
#            ACD.losdist[i] = np.sum(LD.dist0[idxL0]) + np.sum(LD.dist1[idxL1])
#            ACD.loswork[i] = np.sum(LD.work0[idxL0]) + np.sum(LD.work1[idxL1])
#            ACD.severities[i] = np.sum(LD.sev[idxL]) / max(float(len(idxL)),1)
    
    ACD.numpify()    
    
    nCFL = max(len(CD.combi),1)
    nLOS = max(len(LD.combi),1)
    
#    PD.ACD.append(ACD)
    PD.nac.append(len(ACD.id))
    PD.cfl_sum.append(len(CD.combi))
    PD.los_sum.append(len(LD.combi))
    PD.los_count.append(sum(LD.count))
    PD.mcfl.append(sum(ACD.mcfl>1))
    PD.mcfl_max.append(max(ACD.mcfl))
    PD.work.append(np.sum(ACD.work) / float(len(ACD.id)))
    PD.dist.append(np.sum(ACD.dist) / float(len(ACD.id)))
    PD.time.append(np.sum(ACD.t1 - ACD.t0) / float(len(ACD.id)))
    PD.cfltime.append(np.sum(CD.time) / float(nCFL))
    PD.timeincfl.append(np.sum(ACD.cfltime) / float(len(ACD.id)))
    PD.cflwork.append(np.sum(ACD.cflwork) / float(len(ACD.id)))
    PD.cfldist.append(np.sum(ACD.cfldist) / float(len(ACD.id)))
    PD.lostime.append(np.sum(LD.time) / float(nLOS))
    PD.timeinlos.append(np.sum(ACD.lostime) / float(len(ACD.id)))
    PD.loswork.append(np.sum(ACD.loswork) / float(len(ACD.id)))
    PD.losdist.append(np.sum(ACD.losdist) / float(len(ACD.id)))
    PD.severities.append(np.sum(LD.sev) / float(nLOS))
    # Density related stuff
    ind = np.where(np.logical_and(data[:,0] >= 1800,data[:,0] <= 7200))[0]
    dt = np.diff(np.concatenate(([1800.],data[ind[0]:ind[-1]+1,0],[7200.])))
    densA = np.zeros(ind.shape[0]+1)
    densA[1:] = data[ind,22]
    densB = np.zeros(ind.shape[0]+1)
    densB[1:] = data[ind,21]
    if data[0,2][:3] == "CRE":
        densA[0] = densA[1] - 1
        densB[0] = densB[1] - 1
    elif data[0,2][:3] == "DEL":
        densA[0] = densA[1] + 1
        densB[0] = densB[1] + 1
    else:
        densA[0] = densA[1]
        densB[0] = densB[1]
    PD.instreal.append(np.sum(densA*dt)/5400.)        
    PD.density.append(PD.instreal[-1]/Areas[0]*10000.)
    PD.outin.append(np.sum(dt * (densB - densA) / densB) / 5400.)
#    asdasd



PD.numpify()
PD.calcParams()
PD.changeMethodNames()
with open(pdFileName, 'wb') as handle:
    pickle.dump(PD, handle, protocol=pickle.HIGHEST_PROTOCOL)