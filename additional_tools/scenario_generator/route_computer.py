'''
route_computer.py

This function calaculates the route of all aircraft. 

It stores the results in the 'scenario' matrix, which contains the following columns:
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
from aero import nm,ft,kts,vtas2cas
from geo import qdrpos
from check_geo import checkInside, checkOutRect, makeRects


def routeComputer(density, repetition, nacInst, nacTotal, spawnInterval, scenarioDuration, \
               distMin, distMax, altitude, altitudeDel, TASmin, TASmax, gamma, exptLat, exptLon, \
               simLat, simLon, sepMinimum, directory):
                   
    #%% Step 1: Set the random speed based on density and repetition so that 
    #           re-running the scenario generator results in the same scenarios
    
    repetition = repetition + 1
    randomSeed = int(density*repetition)
    np.random.seed(randomSeed)
        
        
    #%% Step 2: Initialize scenario array
        
    # Initialize the scenario array to store the following data. The legend for the 
    # scenario matrix is shown above
    scenario = np.zeros((nacTotal,13))
    
    
    #%% Step 3: AC Spawn times with random addition for all aircraft of this scenario [s]
    
    spawnTimes    = np.linspace(0.0,scenarioDuration*3600.0,num=nacTotal,endpoint=True)
    maxSpawnDelay = spawnInterval
    spawnTimes    = spawnTimes + np.random.uniform(low=0.0,high=maxSpawnDelay,size=nacTotal)
    order         = np.argsort(spawnTimes)
    spawnTimes    = spawnTimes[order]
    scenario[:,0] = spawnTimes
    
    #%% Step 4: Determine the TAS at ground and CAS at cruising and deletion altitude
    
    # Have some extra margin for altitudeDel, make sure that aircraft will descend below original altitudeDel
    altitudeDel = 2.0 * altitudeDel - altitude
    
    # TAS at ground [kts] (at 0 altitude, there is no 'real' difference between CAS and TAS)
    TASground = np.random.uniform(low=TASmin, high=TASmax, size=nacTotal)
    
    # CAS at cruising and deletion altitude needs to be taken into account [kts]
    CAScruise = vtas2cas(TASground*kts, altitude*ft)/kts   
    CASdel = vtas2cas(TASground*kts, altitudeDel*ft)/kts
    
    #%% Step 5: Select Origin and Destination based on heading and distance
    
    # initialize storage lists 
    originLat = np.zeros(nacTotal, dtype=np.float32)
    originLon = np.zeros(nacTotal, dtype=np.float32)
    destLat   = np.zeros(nacTotal, dtype=np.float32)
    destLon   = np.zeros(nacTotal, dtype=np.float32)
    
    # Select a random aircraft heading using a unform random number generator [deg]
    heading   = np.random.uniform(low=0.0, high=360.0, size=nacTotal)
    
    # Select a random distance between origin and destination [NM]
    distance  = np.random.uniform(low=distMin, high=distMax, size=nacTotal)
    
    # Using distance and TAS, find deletion times
    delTimes = distance / TASground * 3600 + spawnTimes
    
    for i in range(int(nacTotal)):
        
        # Make a bool array of existing aircraft at current spawning
        exist = np.logical_and(spawnTimes[i] > spawnTimes, spawnTimes[i] <= delTimes)
        
        # Temp origin lat and lon [deg]. This will have to be re-written
        # if the origin of the experiment is not at (0,0)
        dumOriginLat = np.random.uniform(low=-exptLat, high=exptLat)
        dumOriginLon = np.random.uniform(low=-exptLon, high=exptLon)
        
        # Determine the corresponding temp destination [deg]
        dumDestLat, dumDestLon = qdrpos(dumOriginLat, dumOriginLon, heading[i], distance[i])
        
        # Check if the destination is outside the sim area square
        outside = not checkInside(simLat, simLon, dumDestLat, dumDestLon)
        
        # Calculate current distance of existing aircraft [NM]
        curdistance = TASground[exist] * (spawnTimes[i] - spawnTimes[exist]) / 3600
        
        curLat, curLon = qdrpos(originLat[exist], originLon[exist], heading[exist], curdistance)
        
        # Make rects
        cLat, cLon = makeRects(curLat, curLon, TASground[exist], heading[exist])
        # Check if inside rects
        tooClose = not checkOutRect(cLat, cLon, dumOriginLat, dumOriginLon)
        
        # If destination is outside, or if the origin is too close to previous ones,
        # keep trying different origins until it is not too close and the corresponding
        # destination is inside the sim area. 
        while outside or tooClose:
            
            # try a new origin [deg]
            dumOriginLat = np.random.uniform(low=-exptLat, high=exptLat)
            dumOriginLon = np.random.uniform(low=-exptLon, high=exptLon)
            
            # determin the corresponding destination [deg]
            dumDestLat, dumDestLon = qdrpos(dumOriginLat, dumOriginLon, heading[i], distance[i])
            
            # check is destination is inside
            outside = not checkInside(simLat, simLon, dumDestLat, dumDestLon)
            
            # Check if inside rects
            tooClose = not checkOutRect(cLat, cLon, dumOriginLat, dumOriginLon)
            
        # append the origin and destination lists
        originLat[i] = dumOriginLat
        originLon[i] = dumOriginLon
        destLat[i]   = dumDestLat
        destLon[i]   = dumDestLon
    
    
    #%% Step 6: Determine the Top of Descent lat and lon
    
    # Horizontal distance covered during descent for constant gamma [NM]
    distHorizDescent = ((altitude-altitudeDel)*ft/np.tan(gamma))/nm
    
    # Calculate the bearing from the destination to origin [deg]
    bearingDest2Orig = (np.array(heading)-180.0) % 360.0
    
    # Calculate the latitude and longitude of ToC [deg]
    TODlat, TODlon = qdrpos(destLat, destLon, np.array(bearingDest2Orig), distHorizDescent)
    
    #%% Step 7: Set-up Scenario-matrix
      
    # Store all data into scenario matrix
    scenario[:,1]  = originLat
    scenario[:,2]  = originLon
    scenario[:,3]  = destLat
    scenario[:,4]  = destLon
    scenario[:,5]  = heading
    scenario[:,6]  = distance    
    scenario[:,7]  = CAScruise
    scenario[:,8]  = CASdel
    scenario[:,9]  = altitude
    scenario[:,10] = altitudeDel
    scenario[:,11] = TODlat
    scenario[:,12] = TODlon
    
    #%% Step 8: Pickle dump scenario matrix
    
    # Open the pickle file
    scenName        = "inst%i_rep%i.sp" %(int(nacInst),int(repetition))    
    rawDataFileName = os.path.join(directory, scenName) 
    f               = open(rawDataFileName,"w")
    
    # Dump the data and close the pickle file
    pickle.dump(scenario, f)
    f.close()