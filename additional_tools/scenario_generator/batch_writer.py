'''
batch_writer.py

This script creates simulation batch files. Batch files contain calls to several
scenario files, and specifies simulation settings for each scenario file. This 
way, it is possible to run many simulations in 1 go, without having to start and 
stop each simulation separately. 

Several different batch files are created per setting-file. This ensures that
the run-time per batch is approximately the same.
'''

# import necessary packages
import os
import sys
import numpy as np
from time import strftime, gmtime
from natsort import natsorted

def batchWriter(method, vrange, numDensities, numRepetitions, tMax, scenarioFilesDir):
    
    #%% Step 1: Get the scenario files for this concept and sort them naturally
    scnFiles = [f for f in os.listdir(scenarioFilesDir) if f.count("rep")>0 and f.count(".scn")>0 and f.count("batch")==0]
    setFiles = [f for f in os.listdir(scenarioFilesDir) if f not in scnFiles and f.count("batch")==0]
    scnFiles = natsorted(scnFiles)
    setFiles = natsorted(setFiles)
    
    # santity check
    if len(scnFiles) != numDensities*numRepetitions:
        print "WARNING! Did not find enough scenario files in scenario folder"
        print "Try running this script again with the variable 'recompute = True'" 
        print "Exiting program..."
        sys.exit()
        
    # Reshape scnFiles so that each column contains the repetitions of a particular demand   
    # Remove .T if you want batches with one repetition of all demands instead
    scnFiles = np.reshape(np.array(scnFiles),(numDensities,numRepetitions)).T
    
    
    #%% Step 2: Setup the Super Batch list
    
    # Super Batch list
    superBatch = []
    
    
    # append basic batch settings
    superBatch.append("# ########################################### #\n")
    superBatch.append("# ########################################### #\n")
    superBatch.append("# SUPER BATCH FOR %s_%i-%i\n" %(method, int(vrange[0]), int(vrange[1])))
    superBatch.append("# Number of scn files: %i!\n" %(numRepetitions))
    superBatch.append("# ########################################### #\n")
    superBatch.append("# ########################################### #\n")
    superBatch.append("\n")
    superBatch.append("\n")
    
    # Super Batch name
    superBatchName = "batch_" + method + "_" + str(int(vrange[0])) + "-" + str(int(vrange[1])) + ".scn"
    
    
    #%% Step 3: Loop through scnFiles and create separate batch files for each repetition
    for i in range(scnFiles.shape[0]):
        
        # make a sub array for the O-D scenarios in this batch file
        currentScns = scnFiles[i,:]
        
        # create a name for this batch file
        batchName   = "batch_" + method + "_" + str(int(vrange[0])) + "-" + str(int(vrange[1])) + "_rep" + str(int(i+1)) + ".scn"

        # create lines list to contain all batch scenario lines
        lines = []
        
        # append basic batch settings
        lines.append("# ########################################### #\n")
        lines.append("# Batch: %s\n" %(batchName[:-4]))
        lines.append("# Number of scn files: %i \n" %(numDensities))
        lines.append("# ########################################### #\n")
        lines.append("\n")
        lines.append("\n")
        
        # Loop through the current repetition and make a batch out of all the densities
        for j in range(len(currentScns)):
            lines.append("00:00:00.00>SCEN %s_%i-%i_%s\n" %(method,int(vrange[0]),int(vrange[1]),currentScns[j][:-4]))
            lines.append("00:00:00.00>PCALL %s\n" %(currentScns[j]))
            lines.append("00:00:00.00>PCALL areaDefiniton.scn\n")   
            lines.append("00:00:00.00>PCALL settings_%s_%i-%i.scn\n" %(method, int(vrange[0]), int(vrange[1])) )     
            lines.append("%s>HOLD\n" %(tim2txt(tMax + 60.0)))
            lines.append("\n")
        
        # Extend superBatch list with lines
        superBatch.extend(lines)
        
#        # Write the batch file for the this density
#        g = open(os.path.join(scenarioFilesDir,batchName),"w")    
#        g.writelines(lines)
#        g.close()


    #%% Step 4: Write the super batch to file
    superBatch.append("00:00:00.00>ECHO BATCH DONE\n")
    g = open(os.path.join(scenarioFilesDir,superBatchName),"w")    
    g.writelines(superBatch)
    g.close()

#%% Functions for converting time in seconds to HH:MM:SS.hh
       
def tim2txt(t):
    """Convert time to timestring: HH:MM:SS.hh"""
    return strftime("%H:%M:%S.", gmtime(t)) + i2txt(int((t - int(t)) * 100.), 2)

def i2txt(i, n):
    """Convert integer to string with leading zeros to make it n chars long"""
    itxt = str(i)
    return "0" * (n - len(itxt)) + itxt