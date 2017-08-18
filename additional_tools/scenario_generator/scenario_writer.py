'''
scenario_writer.py

This function converts scenarios to the TrafScript language used by BlueSky

It first loads the scenario computed by route_computer.py, and writes the scenario 
to a text file in the TrafScript language.

The following TrafScript commands are used per aircraft (in the correct order)
1. CRE    -> create the aircraft at cruise altitude, with the right origin, speed and heading 
2. ORIG   -> specify the origin lat/lon for aircraft
5. ADDWPT -> ToD wpt (lat/lon) with speed (cruise) ONLY
6. ADDWPT -> Destination added as a separate waypoint with delation altitude
             to ensure that aircraft don't slow down.  
7. LNAV ON -> Switch on LNAV
8. VNAV ON -> Switch on VNAV

Note: All aircraft are set to be B744

'''

# import necessary packages
import os
import pickle
from time import strftime, gmtime


def scenarioWriter(density,nacInst,repetition,scenarioPicklesDir,scenarioFilesDir):
    
    #%% Constants
    
    # Storage list for scenario definition
    lines = []
    
    # Repetition should not start from 0:
    repetition += 1
    
    # Determine random seed for writing into scenario file
    randomSeed = int(density*repetition)
    
    
    #%% Step 1: Load the scenario file for this concept, nacInst and repetition combo
    
    scenName        = "inst%i_rep%i.sp" %(int(nacInst),int(repetition))    
    directory       = scenarioPicklesDir
    rawDataFileName = os.path.join(directory, scenName) 
    f               = open(rawDataFileName,"r")
    scenario        = pickle.load(f)
    f.close()
    
    
    #%% Step 2: Append scenario header to 'lines' list
    
    lines.append("# ##################################################### #\n")
    lines.append("# Scenario OD Characteristics: \n")
    lines.append("# Density [ac/10,000NM^2]: %2f\n" %(density))
    lines.append("# Num Instantaneous A/C:   %i\n"  % (int(nacInst)))
    lines.append("# Repetition:              %i\n"  %(int(repetition)))
    lines.append("# Random Seed:             %i\n"  %(int(randomSeed)))
    lines.append("# ##################################################### #\n")
    lines.append("\n")
    lines.append("\n")
    lines.append("# Origin-Destination Data:\n")
    
    
    #%% Step 3 Append the route of each aircraft in the scenario to the 'lines' list
    
    for i in range(len(scenario)):
        
        # Convert time to hh:mm:ss format        
        spawnTime = tim2txt(scenario[i,0])
        
        # Call Sign (in accending order)
        callSign = "AC%04d"%(i+1)
        
        # AC type (all aircraft are B747-400)
        acType = "B744"
        
        # Get the remaining variables from scenario for this ac
        originLat = str("%.8f" %(scenario[i,1]))
        originLon = str("%.8f" %(scenario[i,2]))
        destLat   = str("%.8f" %(scenario[i,3]))
        destLon   = str("%.8f" %(scenario[i,4]))
        heading   = str((scenario[i,5]))
        distance  = scenario[i,6]
        CAScruise = str("%.8f" %(scenario[i,7]))
        CASdel    = str("%.8f" %(scenario[i,8]))
        alt       = str("%.8f" %(scenario[i,9]))
        altDel    = str("%.8f" %(scenario[i,10]))
        TODlat    = str("%.8f" %(scenario[i,11]))
        TODlon    = str("%.8f" %(scenario[i,12]))
        
        # construct the scenario lines for this aircraft
        lines.append("\n# %s Direct Distance = %s NM\n" %(callSign,round(distance,2)))
        lines.append(spawnTime + ">CRE," + callSign + "," + acType + "," + originLat + "," + originLon + "," + heading + "," + alt + "," + CAScruise + "\n")
        lines.append(spawnTime + ">ORIG," + callSign + "," +  originLat + "," + originLon + "\n")
        lines.append(spawnTime + ">ADDWPT," + callSign + "," +  TODlat + "," + TODlon + "," + alt + "," + CAScruise + "\n")
        lines.append(spawnTime + ">ADDWPT," + callSign + "," +  destLat + "," + destLon + "," + altDel + "," + CASdel + "\n")
        lines.append(spawnTime + ">LNAV," + callSign + ",ON" + "\n")
        lines.append(spawnTime + ">VNAV," + callSign + ",ON" + "\n")
        
        
    #%% Step 4: Write 'lines' list to Scenario file
    
    # open the file
    scenName    = "inst%i_rep%i.scn" %(int(nacInst),int(repetition))    
    directory   = scenarioFilesDir
    scnFileName = os.path.join(directory, scenName) 
    f           = open(scnFileName,"w")
    
    # write lines and close file
    f.writelines(lines)
    f.close()    

#%% Functions for converting time in seconds to HH:MM:SS.hh
       
def tim2txt(t):
    """Convert time to timestring: HH:MM:SS.hh"""
    return strftime("%H:%M:%S.", gmtime(t)) + i2txt(int((t - int(t)) * 100.), 2)

def i2txt(i, n):
    """Convert integer to string with leading zeros to make it n chars long"""
    itxt = str(i)
    return "0" * (n - len(itxt)) + itxt
