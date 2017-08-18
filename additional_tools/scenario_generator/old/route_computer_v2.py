'''
route_computer.py

This function computes the components of the route

It first loads the O-D matrix for a particular density-repetition combination
and uses this information to compute the cruising CAS, TOD lat and TOD lon.

The resulting scenario is saved as a pickle with the following columns:
0:  spawn time [s]
1:  origin lat [deg]
2:  origin lon [deg]
3:  destination lat [deg]
4:  destination lon [deg]
5:  heading [deg]
6:  horizontal distance [NM]
7:  CAS (cruising) [kts]
8:  CAS (deletion) [kts]
9:  Altitude (cruising) [ft]
10: Altitude (deletion) [ft]
11. TOD lat [deg]
12. TOD lon [deg]

'''

# import necessary packages
import numpy as np
import os
import pickle

# import functions
from aero import ft,nm,kts,vtas2cas
from geo import qdrpos

def routeComputer(nacInst, repetition, altitude, altitudeDel, distMin, distMax, \
                  TASmin, TASmax, gamma, odDirectory, scenarioPicklesDir):
    
    #%% Step 1: Load the appropriate O-D pickle
    repetition += 1
    odFileName  = "inst%i_rep%i.od" %(int(nacInst),int(repetition))   
    odFileName  = os.path.join(odDirectory,odFileName)
    f           = open(odFileName,"r")
    OD          = pickle.load(f)
    f.close()
    
    #%% Step 2: Determine the TAS at ground and CAS at cruising and deletion altitude
    
    # Have some extra margin for altitudeDel, make sure that aircraft will descend below original altitudeDel
    altitudeDel = 2.0 * altitudeDel - altitude
    
    # TAS at ground [kts] (at 0 altitude, there is no 'real' difference between CAS and TAS)
    TASground = np.random.uniform(low=TASmin, high=TASmax,size=len(OD))
    
    # CAS at cruising and deletion altitude needs to be taken into account [kts]
    CAScruise = vtas2cas(TASground*kts, altitude*ft)/kts   
    CASdel = vtas2cas(TASground*kts, altitudeDel*ft)/kts
    
    #%% Step 3: Determine the Top of Descent lat and lon
    
    # Get the heading [deg] from the OD matrix
    heading  = OD[:,5]
    
    # Horizontal distance covered during descent for constant gamma [NM]
    distHorizDescent = ((altitude-altitudeDel)*ft/np.tan(gamma))/nm
    
    # Calculate the bearing from the destination to origin [deg]
    bearingDest2Orig = (np.array(heading)-180.0) % 360.0
    
    # Calculate the latitude and longitude of ToC [deg]
    TODlat, TODlon = qdrpos(OD[:,3], np.array(OD[:,4]), np.array(bearingDest2Orig), distHorizDescent)
    
    
    #%% Step 4: Combine OD and newly calculated route varibles to make 'scenario' array
    
    scenario        = np.zeros((len(OD),13))
    scenario[:,0:7] = OD
    scenario[:,7]   = CAScruise
    scenario[:,8]   = CASdel
    scenario[:,9]   = altitude
    scenario[:,10]  = altitudeDel
    scenario[:,11]  = TODlat
    scenario[:,12]  = TODlon
    
    
    #%% Step 5: Dump sceario matrix to pickle file 
    
    # Open the pickle file
    scenName        = "inst%i_rep%i.sp" %(int(nacInst),int(repetition))    
    directory       = scenarioPicklesDir
    rawDataFileName = os.path.join(directory, scenName) 
    f               = open(rawDataFileName,"w")
    
    # Dump the data and close the pickle file
    pickle.dump(scenario, f)
    f.close()