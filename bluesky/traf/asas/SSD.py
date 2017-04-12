# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:02:02 2017

@author: suthe
"""

from ...tools import geo
from ...tools.aero import ft, nm
import numpy as np
import pickle    
    

def dump(obj):
    filehandler = open('ssd.obj', 'w')
    pickle.dump(obj, filehandler)
    return 'Dumped traf'

try:
    import pyclipper
except ImportError:
    print 'Could not import pyclipper, RESO SSD will not function'

# No idea what this does (placeholder??)
def start(dbconf):
    pass


def detect(dbconf, traf):
    """ Detect all current conflicts """
    
    # Check if ASAS is ON first!    
    if not dbconf.swasas:
        return
    
    # Construct the SSD
    constructSSD(dbconf, traf)

def resolve(dbconf, traf):
    """ Resolve all current conflicts """
    
    # Check if ASAS is ON first!    
    if not dbconf.swasas:
        return
    
    # Initialize SSD variables
    initializeSSD(dbconf, traf)
    
    # Construct the SSD
    constructSSD(dbconf, traf)
    
    # Get resolved speed-vector
    resolve_closest(dbconf, traf)
    
    # Now assign resolutions to variables in the ASAS class
    # No need to cap, since SSD implicitly caps
    dbconf.trk  = np.arctan2(dbconf.resov[:,0], dbconf.resov[:,1]) * 180 / np.pi
    dbconf.spd  = np.sqrt(dbconf.resov[:,0] ** 2 + dbconf.resov[:,1] ** 2)
    dbconf.vs   = traf.vs

def initializeSSD(dbconf, traf):
    """ Initialize variables for SSD """
    # Need to do it here, since ASAS.reset doesn't know ntraf
    dbconf.FRV          = [None] * traf.ntraf
    dbconf.ARV          = [None] * traf.ntraf
    dbconf.inconf       = np.zeros(traf.ntraf, dtype=bool)
    dbconf.resov        = np.zeros((traf.ntraf, 2), dtype=np.float32)
    dbconf.FRV_area     = np.zeros(traf.ntraf, dtype=np.float32)
    dbconf.ARV_area     = np.zeros(traf.ntraf, dtype=np.float32)
    dbconf.confmatrix   = np.zeros((traf.ntraf, traf.ntraf), dtype=bool)

def area(vset):
    # Initialize A as it could be calculated iteratively
    A = 0
    # Check multiple exteriors
    if type(vset[0][0]) == list:
        # Calc every exterior separately
        for i in range(len(vset)):
            A += pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(pyclipper.Area(pyclipper.scale_to_clipper(vset[i]))))
    else:
        # Single exterior
        A = pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(pyclipper.Area(pyclipper.scale_to_clipper(vset))))
    return A

# dbconf is an object of the ASAS class defined in asas.py
def constructSSD(dbconf, traf):
    """ Calculates the FRV and ARV of the SSD """
    output = ''
    # Parameters
    N_angle = 180                   # [-] Number of points on circle (discretization)
    vmin    = dbconf.vmin           # [m/s] Defined in asas.py (100 kts)
    vmax    = dbconf.vmax           # [m/s] Defined in asas.py (600 kts)
    vmin    = 200. / 3600. * nm     # [m/s] Manual definition
    vmax    = 600. / 3600. * nm     # [m/s] Manual definition
    hsep    = dbconf.R              # [m] Horizontal separation (5 NM)
    sepfact = 1.05
    
    # Relevant info from traf
    gsnorth = traf.gsnorth
    gseast  = traf.gseast
    lat     = traf.lat
    lon     = traf.lon
    ntraf   = traf.ntraf
    
    # If no traffic
    if ntraf == 0:
        return
    
    # If only one aircraft
    elif ntraf == 1:
###     # Discretize the circles using points on circle
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
###     # Put points of unit-circle in a (180x2)-array (CW)
        xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
        # Map them into the format ARV wants. Outercircle CCW, innercircle CW
        dbconf.ARV[0] = [list(map(list, np.flipud(xyc * vmax))), list(map(list , xyc * vmin))]
        dbconf.FRV[0] = []
        # Calculate areas and store in dbconf
        dbconf.FRV_area = 0
        dbconf.ARV_area = np.pi * (vmax **2 - vmin ** 2)
        return
        
    # Function qdrdist_matrix needs 4 vectors as input (lat1,lon1,lat2,lon2)
    # To be efficient, calculate all qdr and dist in one function call
    # Example with ntraf = 5:   ind1 = [0,0,0,0,1,1,1,2,2,3]
    #                           ind2 = [1,2,3,4,2,3,4,3,4,4]
    # This way the qdrdist is only calculated once between every aircraft
    # To get all combinations, use this function to get the indices
    ind1, ind2 = qdrdist_matrix_indices(ntraf)
    # Get relative bearing [deg] and distance [nm]
    [qdr, dist] = geo.qdrdist_matrix(lat[ind1], lon[ind1], lat[ind2], lon[ind2])
    # Put result of function from matrix to ndarray
    qdr  = np.reshape(np.array(qdr), np.shape(ind1))
    dist = np.reshape(np.array(dist), np.shape(ind1))
    # SI-units from [deg] to [rad]
    qdr  = np.deg2rad(qdr)
    # Get distance from [nm] to [m]
    dist = dist * nm
    
    # In LoS the VO can't be defined, act as if dist is on edge
    dist[dist < hsep] = hsep
    
    # Calculate vertices of Velocity Obstacle (CCW)
    # These are still in relative velocity space, see derivation in appendix
    # Half-angle of the Velocity obstacle [rad]
    alpha = np.arcsin(hsep * sepfact / dist)
    # Relevant sin/cos/tan
    sinqdr = np.sin(qdr)
    cosqdr = np.cos(qdr)
    tanalpha = np.tan(alpha)
    cosqdrtanalpha = cosqdr * tanalpha
    sinqdrtanalpha = sinqdr * tanalpha
    
    # Relevant x1,y1,x2,y2 (x0 and y0 are zero in relative velocity space)
    x1 = (sinqdr + cosqdrtanalpha) * 2 * vmax
    x2 = (sinqdr - cosqdrtanalpha) * 2 * vmax
    y1 = (cosqdr - sinqdrtanalpha) * 2 * vmax
    y2 = (cosqdr + sinqdrtanalpha) * 2 * vmax
    
    # Use velocity limits for the ring-shaped part
### # Discretize the circles using points on circle
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
### # Put points of unit-circle in a (180x2)-array (CW)
    xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
    # Map them into the format pyclipper wants. Outercircle CCW, innercircle CW
    circle = (tuple(map(tuple, np.flipud(xyc * vmax))), tuple(map(tuple , xyc * vmin)))
    
    # Calculate SSD for every aircraft (See formulas appendix)
    for i in range(ntraf):
        # SSD for aircraft i
        # Get indices that belong to aircraft i
        ind = np.logical_or(ind1 == i,ind2 == i)
        # The i's of the other aircraft
        i_other = np.delete(np.arange(0, ntraf), i)
        # VO from 2 to 1 is mirror of 1 to 2. Only 1 to 2 can be constructed in
        # this manner, so need a correction vector that will mirror the VO
        fix = np.ones(np.shape(i_other))
        fix[i_other < i] = -1
        
        # Get vertices in an x- and y-array of size (ntraf-1)*3x1
        x = np.concatenate((gseast[i_other],
                            x1[ind] * fix + gseast[i_other],
                            x2[ind] * fix + gseast[i_other]))
        y = np.concatenate((gsnorth[i_other],
                            y1[ind] * fix + gsnorth[i_other],
                            y2[ind] * fix + gsnorth[i_other]))
        # Reshape [(ntraf-1)x3] and put arrays in one array [(ntraf-1)x3x2]
        x = np.transpose(x.reshape(3, np.shape(i_other)[0]))
        y = np.transpose(y.reshape(3, np.shape(i_other)[0])) 
        xy = np.dstack((x,y))
        
        # Make a clipper object
        pc = pyclipper.Pyclipper()
        # Add circles (ring-shape) to clipper as subject
        pc.AddPaths(pyclipper.scale_to_clipper(circle), pyclipper.PT_SUBJECT, True)
        # Add each other other aircraft to clipper as clip
        for j in range(np.shape(i_other)[0]):
            VO = pyclipper.scale_to_clipper(map(tuple,xy[j,:,:]))
            pc.AddPath(VO, pyclipper.PT_CLIP, True)
            # Detect conflicts, store in confmatrix
            # Returns 0 if false, -1 if pt is on poly and +1 if pt is in poly.
            if pyclipper.PointInPolygon(pyclipper.scale_to_clipper((gseast[i],gsnorth[i])),VO):
                dbconf.confmatrix[i,i_other[j]] = True
            
        # Execute clipper command
        FRV = pyclipper.scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
        ARV = pyclipper.scale_from_clipper(pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
        # Check multi exteriors, if this layer is not a list, it means it has no exteriors
        if not type(FRV[0][0]) == list:
            FRV = [FRV]
        if not type(ARV[0][0]) == list:
            ARV = [ARV]
        # Store in dbconf 
        dbconf.FRV[i] = FRV
        dbconf.ARV[i] = ARV
        # Calculate areas and store in dbconf
        dbconf.FRV_area[i] = area(FRV)
        dbconf.ARV_area[i] = area(ARV)
        # Temporary
        dbconf.f = dbconf.FRV_area
        dbconf.a = dbconf.ARV_area
    # Update inconf
    dbconf.inconf = np.sum(dbconf.confmatrix, axis=0) > 0
        
#    dump([hsep,vmin,vmax,gsnorth,gseast,lat,lon,ntraf,dbconf.FRV,dbconf.ARV])
    output += 'hoi'
    return output


    

def resolve_closest(dbconf, traf):
    "Calculates closest conflict-free point"
    # It's just linalg, however credits to: http://stackoverflow.com/a/1501725
    # Variables
    ARV     = dbconf.ARV
    gsnorth = traf.gsnorth
    gseast  = traf.gseast
    ntraf   = traf.ntraf
    
    # Loop through SSDs of all aircraft
    for i in range(ntraf):
        # Only those that are in conflict need to resolve
        if dbconf.inconf[i]:
            # Loop through all exteriors and append. Afterwards concatenate
            p = []
            q = []
            for j in range(len(ARV[i])):
                p.append(np.array(ARV[i][j]))
                q.append(np.diff(np.row_stack((p[j], p[j][0])), axis=0))
            p = np.concatenate(p)
            q = np.concatenate(q)
            # Calculate squared distance between edges
            l2 = np.sum(q ** 2, axis=1)
            # Catch l2 == 0 (exception)
            same = l2 < 1e-8
            l2[same] = 1.
            # Calc t
            t = np.sum((np.array([gseast[i], gsnorth[i]]) - p) * q, axis=1) / l2
            # Speed of boolean indices only slightly faster (negligible)
            # t must be limited between 0 and 1
            t = np.clip(t, 0., 1.)
            t[same] = 0.
            # Calculate closest point to each edge
            x = p[:,0] + t * q[:,0]
            y = p[:,1] + t * q[:,1]
            # Get distance squared
            d2 = (x-gseast[i])**2 + (y-gsnorth[i])**2
            # Sort distance
            ind = np.argsort(d2)
            # Store result in dbconf
            dbconf.resov[i,0] = x[ind[0]]
            dbconf.resov[i,1] = y[ind[0]]
    
def check_pyclipper():
    """ Checks whether pyclipper could be imported"""
    if pyclipper:
        return True 
    else:
        return False

def qdrdist_matrix_indices(ntraf):
    """ This function gives the indices that can be used in the lon/lat-vectors """
    # The indices will be n*(n-1)/2 long
    # Only works for n >= 2, which is logical...
    # This is faster than np.triu_indices :)
    tmp_range = np.arange(ntraf - 1, dtype=np.int32)
    ind1 = np.repeat(tmp_range,(tmp_range + 1)[::-1])
    ind2 = np.ones(ind1.shape[0], dtype=np.int32) 
    inds = np.cumsum(tmp_range[1:][::-1] + 1) 
    np.put(ind2, inds, np.arange(ntraf * -1 + 3, 1))
    ind2 = np.cumsum(ind2, out=ind2)
    return ind1, ind2

#import pickle    
#    
#
#def dump(obj):
#    filehandler = open('ssd.obj', 'w')
#    pickle.dump(obj, filehandler)
#    return 'Dumped traf'
#    # Dump
#    dump([hsep,vmin,vmax,gsnorth,gseast,lat,lon,ntraf,dbconf.FRV,dbconf.ARV])
#def test_fun(traf):
#    # CALCULATE EVERYTHING IN SI-UNITS!
#
#    # Horizontal separation [m]
#    hsep = traf.asas.R
#    # Minimum and maximum speeds (set in asas.py) [m/s]
#    vmin = traf.asas.vmin
#    vmax = traf.asas.vmax
#    # North- and East-bound speeds
#    gsnorth = traf.gsnorth
#    gseast = traf.gseast
#    
#    lat1 = traf.lat[0]
#    lon1 = traf.lon[0]
#    lat2 = traf.lat[1:]
#    lon2 = traf.lon[1:]
#    lat = traf.lat
#    lon = traf.lon
#    ntraf = traf.ntraf
#    
#    
#    # Get relative bearing [deg] and distance [nm]
#    [qdr, dist] = geo.qdrdist_matrix(lat1, lon1, lat2, lon2)
#    # Get relative bearing from [deg] to [rad]
#    qdr = np.deg2rad(qdr)
#    # Get distance from [nm] to [m]
#    dist = dist * geo.nm
#    # Get relative x- and y-locations
#    dX = np.multiply(np.sin(qdr), dist)
#    dY = np.multiply(np.cos(qdr), dist)
#    
#    
#    ### Plot
#    # Circles
#    cX = hsep * np.cos(np.arange(0,2*np.pi,np.pi/18))
#    cY = hsep * np.sin(np.arange(0,2*np.pi,np.pi/18))
#    # Creat dump for viewing purposes
#    dump([dX,dY,cX,cY,qdr,dist,hsep,vmin,vmax,gsnorth,gseast,lat,lon,ntraf])
##    fig, ax = plt.subplots()
##    for i in range(len(dX)):
##        plt.plot(cX + dX[0,i],cY + dY[0,i])
##        
##    plt.plot(dX,dY)
##    plt.plot(cX,cY)
##    plt.plot([0],[0])
##    plt.show() 
#    
##    output = vars(traf)
#    return dX,dY