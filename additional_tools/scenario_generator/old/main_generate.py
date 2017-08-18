'''
main_generate.py

This script calls the necessary functions to create 2D scenarios for the 
primary experiment of the thesis (Multi-aircraft conflict resolutions)

Currently the Experiment and Simulation areas are set to be the same.

Computed scenarios are pickled in the data folder

The epxperiment scenarios are .scn files and are stored in the ScenarioFiles 
folder, with subfolders for each airspace concept

Code mostly copied from Emmanuel Sunil's work with few alterations
'''

# import necessary packages
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys

# import functions
from tictoc import tic,toc
from aero import ft,nm,Rearth,kts
from od_computer import ODcomputer
from route_computer import routeComputer
from scenario_writer import scenarioWriter
from settings_writer import settingsWriter
from batch_writer import batchWriter
from area_writer import areaWriter

# Start the clock
tic()

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

# close all Matplotlib figures 
plt.close("all")

# supress silly warnings
import warnings
warnings.filterwarnings("ignore") 

# Welcome Message
print "\033[4mRunning main_generate.py\033[0m"

#%% Inputs

# Use pickled data, or recompute (even if pickle exists)?
recompute = True
countdown = 0

# CR-methods and rules separated by hyphen
methods = ['OFF-OFF','MVP-OFF','SSD-FF1','SSD-FF4']
# V-range in kts
vranges = [[450.0,500.0],[300.0,500.0]]

# Traffic densities and repetitions of each density
minDensity     = 1.00         # [ac/10,000 NM^2]
maxDensity     = 5.0         # [ac/10,000 NM^2]
numDensities   = 4 
numRepetitions = 2

# Average TAS of aircraft [kts]
TASmin = 450.0
TASmax = 500.0
TASavg = (TASmin + TASmax)/2.0

# Minimum Flight Time [hr]
flightTimeMin    = 0.5

# Scenario duration [Hrs] 
scenarioDuration = 2.0    

# Altitude related variables [ft]
alt  = 36000.0

# Horizontal spearation minimum [NM]
sepMinimum = 5.0

# Center of experiment area [deg]
latCenter = 0.0
lonCenter = 0.0

# Altitude to delete descending flights [ft]. Needed for Area definition. 
altDel = 35500.0

# Descend gamma [rad]
gamma = np.arctan2(3000.0*ft,10.0*nm) # 0.049 [rad] = 2.827 [deg]

# Factor to increase the 'sim' area when writing the area definition to reduce 
#   pushing out effect. Only of concern for experiments with CR ON
simAreaFactor = 1.0

# Factor to increase/decerease the area, relative 'expt' area,
# where model parameters are logged
modelAreaFactor = 0.75

#%% Claculated Constants

# common ratio for density steps
commonRatio = np.power(10.0, np.log10(maxDensity/minDensity)/(numDensities-1.0))

# Denities are computed using a geometric series [ac/10,000 NM^2]
densities = minDensity*np.power(commonRatio,range(numDensities))

# Flight distance related variables [NM]

distMin      = flightTimeMin*TASmin
distMax      = flightTimeMin*TASmax
distAvg      = (distMin+distMax)/2.0

# Maximum simulation time (use 0.5h margin)
tMax = (distMax/TASmin + scenarioDuration + 0.5) * 3600.

# Experiment and Simulation area sizing [NM] or [NM^2]
#   Currently Expt and Simulation are set to be the same
sideLengthExpt = 3.0*distMin
sideLengthSim  = 3.0*distMin
areaExpt       = sideLengthExpt**2
areaSim        = sideLengthSim**2

# Number of instantaneous aircraft
#    Divide by 10000 because density is per 10,000 NM^2
nacInst = densities*areaExpt/10000.0

# Spawn rate [1/s] and spawn interval [s]
spawnRate     = (nacInst*TASavg*kts)/(distAvg*nm)
spawnInterval = 1.0/spawnRate

# Total number of aircraft in scenario for the total scenario duration
nacTotal = np.ceil(scenarioDuration*3600.0/spawnInterval)

# Flat earth correction at latCenter
coslatinv = 1.0/np.cos(np.deg2rad(latCenter))

# Corner point of square shaped experiment area. 
#   This code will have to be adjusted if latCenter and LonCenter is not (0,0)
exptLat = latCenter + np.rad2deg(sideLengthExpt*nm/2.0/Rearth) 
exptLon = lonCenter + np.rad2deg(sideLengthExpt*nm/2.0*coslatinv/Rearth) 

# Corner point of square shaped simulation area (epxt area + phantom area)
#   This code will have to be adjusted if latCenter and LonCenter is not (0,0)
simLat = latCenter + np.rad2deg(sideLengthSim*nm/2.0/Rearth) 
simLon = lonCenter + np.rad2deg(sideLengthSim*nm/2.0*coslatinv/Rearth) 


# Storage folder for OD pickles
odDirectory = './data/od'
if not os.path.exists(odDirectory):
    os.makedirs(odDirectory)

# Storage folders for scenario pickles (1 per concept)
scenarioPicklesDir = './data/scenario/'  
if not os.path.exists(scenarioPicklesDir):
     os.makedirs(scenarioPicklesDir)

# Storage folders for experiment scenario and batch files (1 per concept)
scenarioFilesDir = './scenario_files/'
if not os.path.exists(scenarioFilesDir):
     os.makedirs(scenarioFilesDir)

#%% Welcome Message

print "\nThis script computes and generates scenarios for the thesis"
print "\nScenario Files are saved per CR-method in " + scenarioFilesDir
print "\nThe variable 'recompute' is '%s'" %(recompute)
print "This means that scenarios will be recomputed." if recompute else "This means that pickled scenario data will be used to re-write scenario text files"
print "\nYou have %d seconds to cancel (CTRL+C) if 'recompute' should be '%s'..." %(countdown, bool(1-recompute))

# Print Count down!
for i in range(countdown,0,-1):
    print str(i) + "  ", 
    time.sleep(1)

#%% Step 1: Calculate the OD for all densities and repetitions (concept independent)
 
print "\n\n\033[4mStep 1: Computing Origin-Destinations...\033[0m"
for i in range(2,0,-1):
    time.sleep(1)

if recompute:
    
    for i in range(len(densities)):
        
        print "\nDensity %s: %s AC/10000 NM^2" %(i+1, round(densities[i],2))
        
        for rep in range(numRepetitions):
            
            print "Computing OD for Inst: %s, Rep: %s" %(int(nacInst[i]), rep+1)
            
            # Call the ODcomputer function. It will save the OD as a pickle file for 
            # each density-repetition combination.
            ODcomputer(densities[i], rep, nacInst[i], nacTotal[i], spawnInterval[i], \
                        scenarioDuration, distMin, distMax, exptLat, exptLon, \
                        simLat, simLon, sepMinimum, odDirectory)
else:
    # Get the names of the OD pickles
    odPickles = [f for f in os.listdir(odDirectory) if f.count("Rep")>0 and f.count(".od")>0]    
    
    # Sanity check: check if the number of OD pickles is correct
    if len(odPickles) != len(densities)*numRepetitions:
        print "\nWARNING! Did not find enough pickled OD files!" 
        print "Try running this script again with the variable 'recompute = True'"
        print "Exiting program..."
        sys.exit()

#%% Step 2: Calculate the routes for all densities and repetitions

print "\n\n\033[4mStep 2: Computing Routes...\033[0m"
for i in range(2,0,-1):
    time.sleep(1)


if recompute:
    
    for i in range(len(densities)):
    
        print "\nDensity %s: %s AC/10000 NM^2" %(i+1, round(densities[i],2))
        
        for rep in range(numRepetitions):
            
            print "Computing Route for Inst: %s, Rep: %s" %(int(nacInst[i]), rep+1)
            
            # Call the routeComputer function. It will save the scenario as 
            # a pickle file for each density-repetition combination.
            routeComputer(nacInst[i], rep, alt, altDel, distMin, distMax, \
                          TASmin, TASmax, gamma, odDirectory, \
                          scenarioPicklesDir)
else:
    
    # Initialize counter for number of scenarios sanity check
    scnPickles = 0
    
    # Get the names of the OD pickles
    for i in range(len(methods)):
        scnPickles += len([f for f in os.listdir(scenarioPicklesDir+methods[i]) if f.count("Rep")>0 and f.count(".sp")>0])
    
    # Sanity check: check if the number of scenario pickles is correct
    if scnPickles != len(densities)*numRepetitions*len(methods):
        print "\nWARNING! Did not find enough pickled scenario files!" 
        print "Try running this script again with the variable 'recompute = True'"
        print "Exiting program..."
        sys.exit()

#%% Step 3: Write scenario text files

print "\n\n\033[4mStep 3: Writing trafScript scenario text files...\033[0m"
for i in range(2,0,-1):
    time.sleep(1)

for i in range(len(densities)):

    print "\nDensity %s: %s AC/10000 NM^2" %(i+1, round(densities[i],2))
    
    for rep in range(numRepetitions):
        
        print "Writing scenario file for Inst: %s, Rep: %s" %(int(nacInst[i]), rep+1)
        
        # Call the scenarioWriter function to write the text file
        scenarioWriter(densities[i],nacInst[i],rep,scenarioPicklesDir,scenarioFilesDir)

#%% Step 4: Write settings text files

print "\n\n\033[4mStep 4: Writing trafScript settings text files...\n\033[0m"
for i in range(2,0,-1):
    time.sleep(1)

for method in methods:
    
    for vrange in vranges:
        
        print "Writing settings file for CR-Method: %s, V: [%i, %i]" %(method, int(vrange[0]), int(vrange[1]))
        
        # Call the scenarioWriter function to write the text file
        settingsWriter(method,vrange,tMax,scenarioFilesDir)

#%% Step 5: Write Batch files

print "\n\n\033[4mStep 5: Writing trafScript batch text files...\n\033[0m"
for i in range(2,0,-1):
    time.sleep(1)

for method in methods:
    
    for vrange in vranges:
        
        print "Writing superbatch file for CR-Method: %s, V: [%i, %i]" %(method, int(vrange[0]), int(vrange[1]))

        # Call the batchWriter function. It writes 1 batch per method
        batchWriter(method, vrange, numDensities, numRepetitions, tMax, scenarioFilesDir)

#%% Step 6: Write Experiment Area File

print "\n\n\033[4mStep 6: Writing Area Definition File...\n\033[0m"
for i in range(2,0,-1):
    time.sleep(1)
    
print "Writing area definition file"
    
# Call the areaWriter function. It uses trafScript commands to specify the 
# area in which aircraft are allowed to fly.
areaWriter(simLat, simLon, alt + (alt-altDel), altDel, simAreaFactor, scenarioFilesDir)
    
    
#%% Print out the total time taken for generating all the scnearios
print "\n\nScenario Generation Completed!\n"
toc()