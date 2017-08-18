'''
areaWriter.py

This function writes to file the shapes used for the simulation

Three shapes are drawn:
1. SimArea -> if aircraft go outside the SimArea, they are deleted. 
2. SquareModelArea -> Square area where model parameters will be logged
3. CircularModelAreA -> Circular area where model parameters will be logged

All three areas have a top and bottom altitude

As programmed, this function only works properly for areas centered around (0,0)

Note that simlat and simlon are increased by a factor of simAreaFactor to ensure
that the pushing out effect is reduces for CRON cases. 

'''

# import necessary packages
import os

def areaWriter(simLat, simLon, altMax, altDel, simAreaFactor, scenarioFilesDir):
    
    # Initialize list to store trafScript commands
    lines = []
    
    # Header text
    lines.append("# ############################################################## #\n")
    lines.append("# Area Definitions:\n")
    lines.append("#   1. 'SIMAREA'     -> name of aircraft deletion area\n")
    lines.append("# Note: All areas have a 'top' and 'bottom' altitude\n")
    lines.append("#       and slight offsets have been added to prevent probelms\n")
    lines.append("#       due to rounding inaccuracies in BlueSky\n")
    lines.append("# ############################################################## #\n\n")
    
    #%% Step 1: Sim area (Square Shaped)
    
    # Increase the simLat and simLon by simAreaFactor
    simLat = simAreaFactor*simLat
    simLon = simAreaFactor*simLon   
    
    # BlueSky command
    lines.append("00:00:00.00>BOX,SIMAREA" + "," + str(-simLat) + "," + str(-simLon) + "," + str(simLat) + "," + str(simLon) + "," + str(altMax+100.0) +"," + str(altDel)+ "\n")
    lines.append("00:00:00.00>AREA,SIMAREA \n")


    
    #%% Step 4: Write the lines to file 
    
    g = open(os.path.join(scenarioFilesDir,"areaDefiniton.scn"),"w")
    g.writelines(lines)
    g.close()
