'''
processor2.py

This script processes the log-files. Returns an object containing all data
needed for plotting and analysis.
'''

import os
import numpy as np
import pickle
import pandas

# Class processed data
class ProcessedData:
    """ Class holding all processed data """
    
    def __init__(self):
        self.method = []
        self.inst = []
        self.rep = []
        self.vrange = []
        self.cfl_sum = []
        self.los_sum = []
        self.mcfl = []
        self.mcfl_max = []
        self.nac = []
    
    def numpify(self):
        self.method = np.array(self.method)
        self.inst = np.array(self.inst)
        self.rep = np.array(self.rep)
        self.vrange = np.array(self.vrange)
        self.cfl_sum = np.array(self.cfl_sum)
        self.los_sum = np.array(self.los_sum)
        self.mcfl = np.array(self.mcfl)
        self.mcfl_max = np.array(self.mcfl_max)
        self.nac = np.array(self.nac, dtype=np.float32)

class ACData:
    """ Class holding AC-data """
    
    def __init__(self):
        self.id   = []
        self.cfl_idx  = []
        self.los  = []
        self.mcfl = []
        self.t0   = []
        self.t1   = []
        self.work = []
        self.dist = []
    
    def cre(self, acid, t0):
        self.id.append(acid)
        self.t0.append(t0)
        self.t1.append(None)
        self.dist.append(None)
        self.work.append(None)
    
    def dlt(self, idx, t1, dist, work):
        self.t1[idx]   = t1
        self.dist[idx] = dist
        self.work[idx] = work

class CFLLOS:
    """ Class holding CFL and LOS data """
    
    def __init__(self):
        self.combi = []
        self.id0   = []
        self.id1   = []
        self.time  = []
        self.dist0 = []
        self.dist1 = []
        self.work0 = []
        self.work1 = []
        self.open  = []
    
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
    
    def end(self, idx, order, time, dist0, dist1, work0, work1):
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
    elif n == 4:
        print "Combi %s and its ids do not match" %(varA)


# Storage folders for log-files
logFilesDir = './output_inst455/'
#pdFileName  = 'pd1.pickle'

# ProcessedData
PD = ProcessedData()

#%% Step 1: Get the log files
logFiles = [f for f in os.listdir(logFilesDir) if f.count(".log")>0 and f.count("SKYLOG")>0]

#for log in logFiles:
for k in [0]:
    log = logFiles[k]
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
    
    ACD = ACData()
    CD  = CFLLOS()
    skip = False
    for i in range(entries):
        # Sometimes the entry has been processed
        if skip:
            continue
        if data[i,2][:3] == "CRE":
            ACD.cre(data[i,1],data[i,0])
        elif data[i,2][:3] == "DEL":
            try:
                idx = ACD.id.index(data[i,1])
                ACD.dlt(idx, data[i,0], data[i,17], data[i,18])
            except ValueError:
                printError(0, data[i,0])
        elif data[i,2][:3] == "CFL":
            [id0, id1] = sorted([data[i,1],data[i,2][10:]])
            combi = id0 + ' ' + id1
            if data[i,2][4:9] == "START":
                if not data[i+1,2][4:9] == "START":
                    printError(1, i)
                if not combi in CD.combi:
                    # New combi
                    CD.new(combi, id0, id1, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                    skip = True
                else:
                    # Reopen existing
                    idx = CD.combi.index(combi)
            elif data[i,2][4:9] == "ENDED":
                if not data[i+1,2][4:9] == "ENDED":
                    printError(2, i)
                [id0, id1] = sorted([data[i,1],data[i,2][10:]])
                combi = id0 + ' ' + id1
                try:
                    idx = CD.combi.index(combi)
                    if id0 == data[i,1] and id1 == data[i,2][10:]:
                        order = True
                    elif id1 == data[i,1] and id0 == data[i,2][10:]:
                        order = False
                    else:
                        printError(5, combi)
                    if CD.open:
                        CD.end(idx, order, data[i,0], data[i,17], data[i+1,17], data[i,18], data[i+1,18])
                        skip = True
                    else:
                        printError(4, combi)
                except ValueError:
                    printError(3, combi)
                    
                    
            