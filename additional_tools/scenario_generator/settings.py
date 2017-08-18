'''
settings.py

This script calls the settings
'''

from aero import ft,nm,Rearth,kts

# Use pickled data, or recompute (even if pickle exists)?
recompute = True
countdown = 0

# CR-methods and rules separated by hyphen
methods = ['OFF-OFF','MVP-OFF','SSD-FF1','SSD-FF2','SSD-FF3','SSD-FF4','SSD-FF5','SSD-FF6']
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