'''
processor.py

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
        self.cfl  = []
        self.los  = []
        self.mcfl = []
        self.t0   = []
        self.t1   = []
        self.work = []
        self.dist = []
    
    def add(self):
        self.id.append(None)
        self.cfl.append(None)
        self.los.append(None)
        self.mcfl.append(None)
        self.t0.append(None)
        self.t1.append(None)
        self.work.append(None)
        self.dist.append(None)

# Sanity checks
def sanity(data, data_type, log, nAC):
    if not nAC == sum(data[:,2] == 'CRE AC'):
        print log
        print "Not all AC created?"
    if not nAC == sum(data[:,2] == 'DEL AC'):
        print log
        print "Not all AC deleted?"
    if not sum(data_type == 0) == 0:
        print log
        print "Unknown entry encountered"
    if not sum(data_type == 1) == sum(data_type == 2):
        print log
        print "CRE and DEL not equal"
    if not sum(data_type == 3) % 4 == 0:
        print log
        print "CFL not divisible by 4"
    if not sum(data_type == 4) % 4 == 0:
        print log
        print "LOS not divisible by 4"
    return

# Storage folders for log-files
logFilesDir = './output_inst227/'
pdFileName  = 'pd1.pickle'

# ProcessedData
PD = ProcessedData()

#%% Step 1: Get the log files
logFiles = [f for f in os.listdir(logFilesDir) if f.count(".log")>0 and f.count("SKYLOG")>0]


for log in logFiles:
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
    
    # Aircraft
    AC  = np.unique(data[:,1])
    nAC = np.shape(AC)[0]
    
    # For every entry, check what type it is
    data_type = np.zeros(entries, dtype=np.int32)
    conv = {"CRE": 1, "DEL": 2, "CFL": 3, "LOS": 4}
    for i in range(entries):
        data_type[i] = conv[data[i,2][:3]]
    
    # Perform sanity check
    sanity(data, data_type, log, nAC)
    
    # Make array for flying AC
    fly = np.zeros(nAC, dtype=bool)
    
    # Make current conflict-lists
    cfl = [[] for x in xrange(nAC)]
    cfl_hist = [[] for x in xrange(nAC)]
    los = [[] for x in xrange(nAC)]
    los_hist = [[] for x in xrange(nAC)]
    
    cfl_count = np.zeros(nAC, dtype=np.int32)
    los_count = np.zeros(nAC, dtype=np.int32)
    
    # Now loop through each entry and append delete etc.
    for i in range(entries):
        # Register data entry
        idx = int(data[i,1][2:]) - 1
        if data_type[i] == 1:
            if fly[idx]:
                print log
                print data[i,1] + " already created"
            else:
                fly[idx] = True
        elif data_type[i] == 2:
            if not fly[idx]:
                print log
                print data[i,1] + " already deleted or not existent"
            else:
                fly[idx] = False
        elif data_type[i] == 3:
            ido = int(data[i,2][12:]) - 1
            if data[i,2][4:9] == "START":
                if ido in cfl[idx]:
                    print log
                    print str(i) + ": " + data[i,1] + " already in CFL with " + data[i,2][10:]
                    print idx, ido
                    print data[i,2]
                else:
                    cfl[idx].append(ido)
                    # Count concurrent conflicts
                    cfl_count[idx] = max(cfl_count[idx],len(cfl[idx]))
                # Add for history
                if not ido in cfl_hist[idx]:
                    cfl_hist[idx].append(ido)
            elif data[i,2][4:9] == "ENDED":
                if not ido in cfl[idx]:
                    print log
                    print str(i) + ": " + data[i,1] + " not in CFL with " + data[i,2][10:]
                    print idx, ido
                    print data[i,2]
                else:
                    cfl[idx].remove(ido)
            else:
                print "CFL with unknown second arg"
        elif data_type[i] == 4:
            ido = int(data[i,2][12:]) - 1
            if data[i,2][4:9] == "START":
                if ido in los[idx]:
                    print log
                    print str(i) + ": " + data[i,1] + " already in LOS with " + data[i,2][10:]
                    print idx, ido
                    print data[i,2]
                else:
                    los[idx].append(ido)
                    # Count concurrent LoSs
                    los_count[idx] = max(los_count[idx],len(los[idx]))
                # Add for history
                if not ido in los_hist[idx]:
                    los_hist[idx].append(ido)
            elif data[i,2][4:9] == "ENDED":
                if not ido in los[idx]:
                    print log
                    print str(i) + ": " + data[i,1] + " not in LOS with " + data[i,2][10:]
                    print idx, ido
                    print data[i,2]
                else:
                    los[idx].remove(ido)
            else:
                print "LOS with unknown second arg"
    
        # Process current data
        else:
            print "Sanity failed"
    
    # Process cfl        
    cfl_sum = 0
    for i in range(nAC):
        cfl_sum += len(cfl_hist[i])
    if not cfl_sum % 2 == 0:
        print "CFL_sum not divisble by 2"
    cfl_sum /= 2
    # Process los        
    los_sum = 0
    for i in range(nAC):
        los_sum += len(los_hist[i])
    if not los_sum % 2 == 0:
        print "LOS_sum not divisble by 2"
    los_sum /= 2
    
    PD.method.append(method)
    PD.inst.append(inst)
    PD.rep.append(rep)
    PD.vrange.append(vrange)
    PD.cfl_sum.append(cfl_sum)
    PD.los_sum.append(los_sum)
    PD.mcfl.append(sum(cfl_count>1))
    PD.mcfl_max.append(max(cfl_count))
    PD.nac.append(nAC)


with open(pdFileName, 'wb') as handle:
    pickle.dump(PD, handle, protocol=pickle.HIGHEST_PROTOCOL)