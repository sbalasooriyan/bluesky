# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:02:02 2017

@author: suthe
"""

from ...tools import geo
import numpy as np
import pickle    
    

def dump(obj):
    filehandler = open('ssd.obj', 'w')
    pickle.dump(obj, filehandler)
    return 'Dumped traf'

def test_fun(traf):
    # CALCULATE EVERYTHING IN SI-UNITS!

    # Horizontal separation [m]
    hsep = traf.asas.R
    # Minimum and maximum speeds (set in asas.py) [m/s]
    vmin = traf.asas.vmin
    vmax = traf.asas.vmax
    # North- and East-bound speeds
    gsnorth = traf.gsnorth
    gseast = traf.gseast
    
    lat1 = traf.lat[0]
    lon1 = traf.lon[0]
    lat2 = traf.lat[1:]
    lon2 = traf.lon[1:]
    lat = traf.lat
    lon = traf.lon
    ntraf = traf.ntraf
    
    
    # Get relative bearing [deg] and distance [nm]
    [qdr, dist] = geo.qdrdist_matrix(lat1, lon1, lat2, lon2)
    # Get relative bearing from [deg] to [rad]
    qdr = np.deg2rad(qdr)
    # Get distance from [nm] to [m]
    dist = dist * geo.nm
    # Get relative x- and y-locations
    dX = np.multiply(np.sin(qdr), dist)
    dY = np.multiply(np.cos(qdr), dist)
    
    
    ### Plot
    # Circles
    cX = hsep * np.cos(np.arange(0,2*np.pi,np.pi/18))
    cY = hsep * np.sin(np.arange(0,2*np.pi,np.pi/18))
    # Creat dump for viewing purposes
    dump([dX,dY,cX,cY,qdr,dist,hsep,vmin,vmax,gsnorth,gseast,lat,lon,ntraf])
#    fig, ax = plt.subplots()
#    for i in range(len(dX)):
#        plt.plot(cX + dX[0,i],cY + dY[0,i])
#        
#    plt.plot(dX,dY)
#    plt.plot(cX,cY)
#    plt.plot([0],[0])
#    plt.show() 
    
#    output = vars(traf)
    return dX,dY