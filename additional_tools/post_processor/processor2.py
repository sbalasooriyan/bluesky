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

warnings.simplefilter('always')

# Class processed data
class ProcessedData:
    """ Class holding all processed data """
    
    def __init__(self):
        self.method     = []
        self.inst       = []
        self.rep        = []
        self.vrange     = []
        self.cfl_sum    = []
        self.los_sum    = []
        self.mcfl       = []
        self.mcfl_max   = []
        self.nac        = []
#        self.ACD        = []
        self.time       = []
        self.work       = []
        self.dist       = []
        self.cfltime    = []
        self.cflwork    = []
        self.cfldist    = []
        self.lostime    = []
        self.loswork    = []
        self.losdist    = []
        self.severities = []
        self.instreal   = []
        self.density    = []
        self.outin      = []
    
    def numpify(self):
        self.method     = np.array(self.method)
        self.inst       = np.array(self.inst)
        self.rep        = np.array(self.rep)
        self.vrange     = np.array(self.vrange)
        self.cfl_sum    = np.array(self.cfl_sum)
        self.los_sum    = np.array(self.los_sum)
        self.mcfl       = np.array(self.mcfl)
        self.mcfl_max   = np.array(self.mcfl_max, dtype=np.float32)
        self.nac        = np.array(self.nac, dtype=np.float32)
        self.time       = np.array(self.time, dtype=np.float32)
        self.work       = np.array(self.work, dtype=np.float32)
        self.dist       = np.array(self.dist, dtype=np.float32)
        self.cfltime    = np.array(self.cfltime, dtype=np.float32)
        self.cflwork    = np.array(self.cflwork, dtype=np.float32)
        self.cfldist    = np.array(self.cfldist, dtype=np.float32)
        self.lostime    = np.array(self.lostime, dtype=np.float32)
        self.loswork    = np.array(self.loswork, dtype=np.float32)
        self.losdist    = np.array(self.losdist, dtype=np.float32)
        self.severities = np.array(self.severities, dtype=np.float32)
        self.instreal   = np.array(self.instreal, dtype=np.float32)
        self.density    = np.array(self.density, dtype=np.float32)
        self.outin      = np.array(self.outin, dtype=np.float32)

    def calcDEP(self):
        self.dep = np.zeros(np.shape(self.method), dtype=np.float32)
        for i in range(np.shape(self.method)[0]):
            if self.method[i] != "OFF-OFF":
                ind = np.where(np.logical_and(self.method == "OFF-OFF",self.inst == self.inst[i], self.rep == self.rep[i]))[0]
                if np.shape(ind)[0] > 0:
                    self.dep[i] = float(self.cfl_sum[i]) / self.cfl_sum[ind[0]] - 1
                else:
                    print "Doesn't have OFF-OFF part for %s with inst = %d and rep %d" %(self.method[i], self.inst[i], self.rep[i])

class ACData:
    """ Class holding AC-data """
    
    def __init__(self):
        self.id         = []
        self.cfl        = []
        self.los        = []
        self.mcfl       = []
        self.t0         = []
        self.t1         = []
        self.work       = []
        self.dist       = []
        self.cfl_cur    = []
        self.cfltime    = []
        self.lostime    = []
        self.cfldist    = []
        self.cflwork    = []
        self.losdist    = []
        self.loswork    = []
        self.severities = []
    
    def cre(self, acid, t0):
        self.id.append(acid)
        self.t0.append(t0)
        self.t1.append(None)
        self.dist.append(None)
        self.work.append(None)
        self.cfl.append(None)
        self.los.append(None)
        self.mcfl.append(0)
        self.cfl_cur.append(0)
        self.cfltime.append(0)
        self.cfldist.append(0)
        self.cflwork.append(0)
        self.lostime.append(0)
        self.losdist.append(0)
        self.loswork.append(0)
        self.severities.append(np.array([]))
    
    def dlt(self, idx, t1, dist, work):
        self.t1[idx]   = t1
        self.dist[idx] = dist
        self.work[idx] = work
    
    def numpify(self):
        self.id         = np.array(self.id)
        self.cfl        = np.array(self.cfl)
        self.los        = np.array(self.los)
        self.mcfl       = np.array(self.mcfl)
        self.t0         = np.array(self.t0)
        self.t1         = np.array(self.t1)
        self.work       = np.array(self.work)
        self.dist       = np.array(self.dist)
        self.cfl_cur    = np.array(self.cfl_cur)
        self.cfltime    = np.array(self.cfltime)
        self.lostime    = np.array(self.lostime)
        self.cfldist    = np.array(self.cfldist)
        self.cflwork    = np.array(self.cflwork)
        self.losdist    = np.array(self.losdist)
        self.loswork    = np.array(self.loswork)
        self.severities = np.array(self.severities)

class CFLLOS:
    """ Class holding CFL and LOS data """
    
    def __init__(self, LOS=False):
        self.combi = []
        self.id0   = []
        self.id1   = []
        self.time  = []
        self.dist0 = []
        self.dist1 = []
        self.work0 = []
        self.work1 = []
        self.open  = []
        if LOS:
            self.los = True
            self.sev = []
        else:
            self.los = False
    
    def new(self, combi, id0, id1, time, dist0, dist1, work0, work1):
        self.combi.append(combi)
        self.id0.append(id0)
        self.id1.append(id1)
        self.time.append(time)
        self.work0.append(work0)
        self.work1.append(work1)
        self.dist0.append(dist0)
        self.dist1.append(dist1)
        self.open.append(True)
        if self.los:
            self.sev.append(0)
    
    def end(self, idx, order, time, dist0, dist1, work0, work1, sev=None):
        self.time[idx] = time - self.time[idx]
        self.open[idx] = False
        if order:
            self.dist0[idx] = dist0 - self.dist0[idx]
            self.work0[idx] = work0 - self.work0[idx]
            self.dist1[idx] = dist1 - self.dist1[idx]
            self.work1[idx] = work1 - self.work1[idx]
        else:
            self.dist0[idx] = dist0 - self.dist1[idx]
            self.work0[idx] = work0 - self.work1[idx]
            self.dist1[idx] = dist1 - self.dist0[idx]
            self.work1[idx] = work1 - self.work0[idx]
        if self.los:
            self.sev[idx] = max(self.sev[idx], sev)
    
    def cnt(self, idx, order, time, dist0, dist1, work0, work1):
        self.time[idx] = time - self.time[idx]
        self.open[idx] = True
        if order:
            self.dist0[idx] = dist0 - self.dist0[idx]
            self.work0[idx] = work0 - self.work0[idx]
            self.dist1[idx] = dist1 - self.dist1[idx]
            self.work1[idx] = work1 - self.work1[idx]
        else:
            self.dist0[idx] = dist0 - self.dist1[idx]
            self.work0[idx] = work0 - self.work1[idx]
            self.dist1[idx] = dist1 - self.dist0[idx]
            self.work1[idx] = work1 - self.work0[idx]
    
    def numpify(self):
        self.combi = np.array(self.combi)
        self.id0   = np.array(self.id0)
        self.id1   = np.array(self.id1)
        self.time  = np.array(self.time)
        self.dist0 = np.array(self.dist0)
        self.dist1 = np.array(self.dist1)
        self.work0 = np.array(self.work0)
        self.work1 = np.array(self.work1)
        self.open  = np.array(self.open)
        if self.los:
            self.sev = np.array(self.sev)
        
            
    
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
logFilesDir = './check/'
pdFileName  = logFilesDir[2:-1] + '.pickle'
Areas       = [455625.0, 810000.0]

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
    # Vrange in integers
    vrange = map(int, vrange.split("-"))
    # Inst
    inst = int(inst[4:])
    # Repetition
    rep = int(rep[3:])
    
    # Read log-file and store in data
    data = pandas.read_csv(logFilesDir+log, comment='#', delimiter=',', header=None)
    data = data.as_matrix()
    entries = np.shape(data)[0]
    
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
                if not combi in LD.combi:
                    # New combi
                    LD.new(combi, id0, id1, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                    skip = True
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
                    else:
                        printError(6, combi)
            elif data[i,2][4:9] == "ENDED":
                if not data[i+1,2][4:9] == "ENDED":
                    printError(2, i)
                [id0, id1] = sorted([data[i,1],data[i,2][10:]])
                combi = id0 + ' ' + id1
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
                    else:
                        printError(4, combi)
                except ValueError:
                    printError(3, combi)
    
    
    CD.numpify()
    LD.numpify()
    
    for i in range(len(ACD.id)):
        # Bools
        idxC0 = ACD.id[i] == CD.id0
        idxC1 = ACD.id[i] == CD.id1
        idxC = np.where(np.logical_or(idxC0, idxC1))[0]
        
        idxL0 = np.in1d(LD.id0, ACD.id[i])
        idxL1 = np.in1d(LD.id1, ACD.id[i])
        idxL = np.where(np.logical_or(idxL0, idxL1))[0]
        # Fill CFLs
        ACD.cfl[i] = idxC
        # Fill LOSs
        ACD.los[i] = idxL
        # Other stuff
        if len(CD.combi) > 0:
            ACD.cfltime[i] = np.sum(CD.time[idxC])
            ACD.cfldist[i] = np.sum(CD.dist0[idxC0]) + np.sum(CD.dist1[idxC1])
            ACD.cflwork[i] = np.sum(CD.work0[idxC0]) + np.sum(CD.work1[idxC1])
        if len(LD.combi) > 0:
            ACD.lostime[i] = np.sum(LD.time[idxL])
            ACD.losdist[i] = np.sum(LD.dist0[idxL0]) + np.sum(LD.dist1[idxL1])
            ACD.loswork[i] = np.sum(LD.work0[idxL0]) + np.sum(LD.work1[idxL1])
            ACD.severities[i] = np.sum(LD.sev[idxL]) / max(float(len(idxL)),1)
    
    ACD.numpify()    
    
    nCFL = max(len(CD.combi),1)
    nLOS = max(len(LD.combi),1)
    
#    PD.ACD.append(ACD)
    PD.nac.append(len(ACD.id))
    PD.cfl_sum.append(len(CD.combi))
    PD.los_sum.append(len(LD.combi))
    PD.mcfl.append(sum(ACD.mcfl>1))
    PD.mcfl_max.append(max(ACD.mcfl))
    PD.work.append(np.sum(ACD.work) / float(len(ACD.id)))
    PD.dist.append(np.sum(ACD.dist) / float(len(ACD.id)))
    PD.time.append(np.sum(ACD.t1 - ACD.t0) / float(len(ACD.id)))
    PD.cfltime.append(np.sum(ACD.cfltime) / float(nCFL))
    PD.cflwork.append(np.sum(ACD.cflwork) / float(nCFL))
    PD.cfldist.append(np.sum(ACD.cfldist) / float(nCFL))
    PD.lostime.append(np.sum(ACD.lostime) / float(nLOS))
    PD.loswork.append(np.sum(ACD.loswork) / float(nLOS))
    PD.losdist.append(np.sum(ACD.losdist) / float(nLOS))
    PD.severities.append(np.sum(ACD.severities) / float(nLOS))
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
PD.calcDEP()
with open(pdFileName, 'wb') as handle:
    pickle.dump(PD, handle, protocol=pickle.HIGHEST_PROTOCOL)