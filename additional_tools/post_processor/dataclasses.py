import numpy as np
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
        self.los_count  = []
        self.mcfl       = []
        self.mcfl_max   = []
        self.nac        = []
#        self.ACD        = []
        self.time       = []
        self.work       = []
        self.dist       = []
        self.cfltime    = []
        self.timeincfl  = []
        self.cflwork    = []
        self.cfldist    = []
        self.lostime    = []
        self.timeinlos  = []
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
        self.los_count  = np.array(self.los_count)
        self.mcfl       = np.array(self.mcfl)
        self.mcfl_max   = np.array(self.mcfl_max, dtype=np.float32)
        self.nac        = np.array(self.nac, dtype=np.float32)
        self.time       = np.array(self.time, dtype=np.float32)
        self.work       = np.array(self.work, dtype=np.float32)
        self.dist       = np.array(self.dist, dtype=np.float32)
        self.cfltime    = np.array(self.cfltime, dtype=np.float32)
        self.timeincfl  = np.array(self.timeincfl, dtype=np.float32)
        self.cflwork    = np.array(self.cflwork, dtype=np.float32)
        self.cfldist    = np.array(self.cfldist, dtype=np.float32)
        self.lostime    = np.array(self.lostime, dtype=np.float32)
        self.timeinlos  = np.array(self.timeinlos, dtype=np.float32)
        self.loswork    = np.array(self.loswork, dtype=np.float32)
        self.losdist    = np.array(self.losdist, dtype=np.float32)
        self.severities = np.array(self.severities, dtype=np.float32)
        self.instreal   = np.array(self.instreal, dtype=np.float32)
        self.density    = np.array(self.density, dtype=np.float32)
        self.outin      = np.array(self.outin, dtype=np.float32)

    def calcParams(self):
        self.dep = np.zeros(np.shape(self.method), dtype=np.float32)
        self.ipr = 1 - np.float32(self.los_sum) / np.float32(self.cfl_sum)
        self.diste = np.zeros(np.shape(self.method), dtype=np.float32)
        self.worke = np.zeros(np.shape(self.method), dtype=np.float32)
        self.timee = np.zeros(np.shape(self.method), dtype=np.float32)
        self.densitye = np.zeros(np.shape(self.method), dtype=np.float32)
        for i in range(np.shape(self.method)[0]):            
            if self.method[i] != "OFF-OFF":
                ind = np.where(np.logical_and(self.method == "OFF-OFF",self.inst == self.inst[i], self.rep == self.rep[i]))[0]
                if np.shape(ind)[0] > 0:
                    self.dep[i] = float(self.cfl_sum[i]) / self.cfl_sum[ind[0]] - 1
                    self.diste[i] = float(self.dist[i]) / self.dist[ind[0]] - 1
                    self.worke[i] = float(self.work[i]) / self.work[ind[0]] - 1
                    self.timee[i] = float(self.time[i]) / self.time[ind[0]] - 1
                    self.densitye[i] = float(self.density[i]) / self.density[ind[0]] - 1
                else:
                    print "Doesn't have OFF-OFF part for %s with inst = %d and rep %d" %(self.method[i], self.inst[i], self.rep[i])

    def changeMethodNames(self):
        old = ["MVP-OFF","OFF-OFF","SSD-FF1","SSD-FF2","SSD-FF3","SSD-FF4","SSD-FF5","SSD-FF6","SSD-FF8","SSD-FF9"]
        new = ["MVP","NO CR","RS1","RS2","RS6","RS5","RS3","RS4","RS7","RS8"]
        for i in range(len(old)):
            self.method[self.method == old[i]] = new[i]

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
        self.los_cur    = []
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
        self.los_cur.append(0)
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
        self.los_cur    = np.array(self.los_cur)
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
            self.count = []
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
            self.count.append(True)
    
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