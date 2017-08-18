'''
ODcomputer.py

This function calaculates the horizontal trajectory of aircraft. 

It stores the results in the 'OD' matrix, which contains the following columns: 
0: spawn time [s]
1: origin lat [deg]
2: orign lon[deg]
3: destination lat [deg]
4: destination lon [deg]
5: heading [deg]
6: horizontal distance [NM]

'''

# import necessary packages
import numpy as np
import os
import pickle

# import functions
from aero import nm
from geo import qdrpos,latlondist
from check_inside import checkInside


def ODcomputer(density, repetition, nacInst, nacTotal, spawnInterval, scenarioDuration, \
               distMin, distMax, exptLat, exptLon, simLat, simLon, sepMinimum, directory):
                   
    #%% Step 1: Set the random speed based on density and repetition so that 
    #           re-running the scenario generator results in the same scnearios
    
    repetition = repetition + 1
    randomSeed = int(density*repetition)
    np.random.seed(randomSeed)
        
        
    #%% Step 2: Initialize OD array
        
    # Initialize the OD array to store the following data. The legend for the 
    # OD matrix is shown above
    OD = np.zeros((nacTotal,7))
    
    
    #%% Step 3: AC Spawn times with random addition for all aircraft of this scenario [s]
    
    spawnTimes    = np.linspace(0.0,scenarioDuration*3600.0,num=nacTotal,endpoint=True)
    maxSpawnDelay = spawnInterval
    spawnTimes    = spawnTimes + np.random.uniform(low=0.0,high=maxSpawnDelay,size=nacTotal)
    order         = np.argsort(spawnTimes)
    spawnTimes    = spawnTimes[order]
    OD[:,0]       = spawnTimes
    
    
    #%% Step 4: Select Origin and Destination based on heading and distance

    # initialize storage lists 
    originLat = []
    originLon = []
    destLat   = []
    destLon   = []
    heading   = []
    distance  = []
    
    for i in range(int(nacTotal)):
        
        # Select a random aircraft heading using a unform random number generator [deg]
        direction = np.random.uniform(low=0.0, high=360.0)
        heading.append(direction)
        
        # Select a random distance between origin and destination [NM]
        dist = np.random.uniform(low=distMin, high=distMax)
        distance.append(dist)
        
        # Temp origin lat and lon [deg]. This will have to be re-written
        # if the origin of the experiment is not at (0,0)
        tempOriginLat = np.random.uniform(low=-exptLat, high=exptLat)
        tempOriginLon = np.random.uniform(low=-exptLon, high=exptLon)
        
        # Determine the corresponding temp destination [deg]
        tempDestLat, tempDestLon = qdrpos(tempOriginLat, tempOriginLon, direction, dist)
        
        # Check if the destination is outside the sim area square
        outside = not checkInside(simLat, simLon, tempDestLat, tempDestLon)
        
        # Determine the distance of proposed origin to the previous nacInst origins [NM]
        dist2previousOrigins = latlondist(np.array(originLat[-int(nacInst):]), np.array(originLon[-int(nacInst):]), \
                                 np.array(tempOriginLat), np.array(tempOriginLon))/nm
        
        # Check if the proposed origin is too close to any of the previous nacInst origins
        tooCloseOrigins = len(dist2previousOrigins[dist2previousOrigins<sepMinimum])>0
        
        # Determine the distance of proposed destination to the previous nacInst destinations [NM]
        dist2previousDests = latlondist(np.array(destLat[-int(nacInst):]), np.array(destLon[-int(nacInst):]), \
                       np.array(tempDestLat), np.array(tempDestLon))/nm
                         
        # Check if the proposed destination is too close to any of the previous nacInst destinations
        tooCloseDestinations = len(dist2previousDests[dist2previousDests<sepMinimum])>0
        
        tooClose = tooCloseOrigins or tooCloseDestinations
        
        # If destination is outside, or if the origin is too close to previous ones,
        # or if the destination is too close to a previous ones, then
        # keep trying different origins until it is not too close and the corresponding
        # destination is inside the sim area. 
        while outside or tooClose:
            
            # try a new origin [deg]
            tempOriginLat = np.random.uniform(low=-exptLat, high=exptLat)
            tempOriginLon = np.random.uniform(low=-exptLon, high=exptLon)
            
            # determin the corresponding destination [deg]
            tempDestLat, tempDestLon = qdrpos(tempOriginLat, tempOriginLon, direction, dist)
            
            # check is destination is inside
            outside = not checkInside(simLat, simLon, tempDestLat, tempDestLon)
            
            # Determine the distance of proposed origin to the previous nacInst origins [NM]
            dist2previousOrigins = latlondist(np.array(originLat[-int(nacInst):]), np.array(originLon[-int(nacInst):]), \
                                   np.array(tempOriginLat), np.array(tempOriginLon))/nm
                                     
            # Check if the proposed origin is too close to any of the previous nacInst origins
            tooCloseOrigins = len(dist2previousOrigins[dist2previousOrigins<sepMinimum])>0
            
            # Determine the distance of proposed destination to the previous nacInst destinations [NM]
            dist2previousDests = latlondist(np.array(destLat[-int(nacInst):]), np.array(destLon[-int(nacInst):]), \
                                   np.array(tempDestLat), np.array(tempDestLon))/nm
                                     
            # Check if the proposed destination is too close to any of the previous nacInst destinations
            tooCloseDestinations = len(dist2previousDests[dist2previousDests<sepMinimum])>0
            
            
            tooClose = tooCloseOrigins or tooCloseDestinations
            
        # append the origin and destination lists
        originLat.append(tempOriginLat)
        originLon.append(tempOriginLon)
        destLat.append(tempDestLat)
        destLon.append(tempDestLon)
        
    # Store all data into scenario matrix
    OD[:,1] = np.array(originLat)
    OD[:,2] = np.array(originLon)
    OD[:,3] = np.array(destLat)
    OD[:,4] = np.array(destLon)
    OD[:,5] = np.array(heading)
    OD[:,6] = np.array(distance)
    
    
    #%% Step 5: Pickle dump OD matrix
    
    # Open the pickle file
    scenName        = "inst%i_rep%i.od" %(int(nacInst),int(repetition))    
    rawDataFileName = os.path.join(directory, scenName) 
    f               = open(rawDataFileName,"w")
    
    # Dump the data and close the pickle file
    pickle.dump(OD, f)
    f.close()