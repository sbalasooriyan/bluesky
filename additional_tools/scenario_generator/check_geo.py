'''
checkInside.py

Check if point (lat,lon) is inside the square shaped area defined by 
 corner points (-refLat,-refLon) and (refLat,refLon)
 
All lat and lon in DEG


'''
from geo import qdrpos,latlondist
import numpy as np

def checkInside(refLat,refLon,lat,lon):
    
    inside = ((-refLat <= lat) & (lat <= refLat)) & \
             ((-refLon <= lon) & (lon <= refLon))
             
    return inside

def makeRects(curLat, curLon, TAS, heading):
    # Number of aircraft
    nAC = heading.shape[0]
    # Rectangle half-height [NM]
    spacing = 6.0
    # Look-aheadtime rectangle [h]
    look = 5. / 60.
    
    # 'Diagonals'
    distA  = (spacing * spacing + TAS * look * TAS * look) ** 0.5
    distB  = (spacing * spacing * 2) ** 0.5
    
    # Head-change
    dhead = np.zeros((nAC, 4), dtype=np.float32)
    dhead[:,0] = np.mod(heading - np.degrees(np.arctan2(spacing,distA)) + 360., 360.)
    dhead[:,1] = np.mod(heading + np.degrees(np.arctan2(spacing,distA)) + 360., 360.)
    dhead[:,2] = np.mod(heading + 135. + 360., 360.)
    dhead[:,3] = np.mod(heading - 135. + 360., 360.)
    
    # Rectangle corners (only three needed)
    cLat = np.zeros((nAC, 3), dtype=np.float32)
    cLon = np.zeros((nAC, 3), dtype=np.float32)
    cLat[:,0], cLon[:,0] = qdrpos(curLat, curLon, dhead[:,0], distA)
    cLat[:,1], cLon[:,1] = qdrpos(curLat, curLon, dhead[:,1], distA)
    cLat[:,2], cLon[:,2] = qdrpos(curLat, curLon, dhead[:,2], distB)
    
    return cLat, cLon
    

def checkOutRect(cLat, cLon, oriLat, oriLon):
    # Number of aircraft
    nAC = cLat.shape[0]
    
    # Point in rect: https://math.stackexchange.com/a/190373
    for i in range(nAC):
        # Set up vectors
        AB = np.array([cLon[i,1] - cLon[i,0], cLat[i,1] - cLat[i,0]])
        AM = np.array([oriLon    - cLon[i,0], oriLat    - cLat[i,0]])
        AD = np.array([cLon[i,2] - cLon[i,0], cLat[i,2] - cLat[i,0]])
        
        AMAB = np.dot(AM,AB)
        AMAD = np.dot(AM,AD)
        ABAB = np.dot(AB,AB)
        ADAD = np.dot(AD,AD)
        
        # Check inside rect
        if 0 < AMAB and AMAB < ABAB and 0 < AMAD and AMAD < ADAD:
            return False

    # Didn't get inside, thus outside
    return True