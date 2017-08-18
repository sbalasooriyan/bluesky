'''
settings_writer.py

This function converts setting-parameters to the TrafScript language used by BlueSky

It writes the settings the scenario to a text file in the TrafScript language.

The following parts define the settings:
1. Basic Settings
2. ASAS Settings
3. Log Settings
'''

# import necessary packages
import os
from time import strftime, gmtime


def settingsWriter(method,vrange,tMax,settingsFilesDir):
    
    #%% Constants
    
    # Split method into reso and rule
    reso,rule = method.split("-")
    
    # Storage list for scenario definition
    lines = []
    
    # Times
    tZero = tim2txt(0)
    tMax  = tim2txt(tMax)
    
    
    #%% Step 1: Append settings header to 'lines' list
    
    lines.append("# ##################################################### #\n")
    lines.append("# Settings Parameters: \n")
    lines.append("# CR-Method:  %s\n" %(reso))
    lines.append("# Priorule:   %s\n" %(rule))
    lines.append("# V-range:    %i-%i\n" %(int(vrange[0]),int(vrange[1])))
    lines.append("# ##################################################### #\n")
    lines.append("\n")
    lines.append("\n")

    #%% Step 2: Append basic settings
    lines.append("# Basic Settings:\n")
    lines.append(tZero + ">PAN 0.0, 0.0\n")
    lines.append(tZero + ">ZOOM 0.15\n")
    lines.append(tZero + ">FF\n")
    lines.append(tZero + ">DT 0.05\n")
    lines.append("\n")
    lines.append("\n")

    #%% Step 3: Append asas settings
    lines.append("# ASAS Settings:\n")
    lines.append(tZero + ">ASAS ON\n")
    lines.append(tZero + ">RESO " + reso + "\n")
    lines.append(tZero + ">PRIORULES ")
    if rule == "OFF":
        lines.append("OFF\n")
    else:
        lines.append("ON " + rule + "\n")
    lines.append(tZero + ">DT 0.05\n")
    lines.append(tZero + ">ASASV MIN " + str(vrange[0]) + "\n")
    lines.append(tZero + ">ASASV MAX " + str(vrange[1]) + "\n")
    lines.append("\n")
    lines.append("\n")

    #%% Step 4: Append log settings
#    lines.append(tZero + ">SNAPLOG ON\n")
    lines.append(tZero + ">CFLLOG ON\n")
#    lines.append(tZero + ">EVTLOG ON\n")
    lines.append(tZero + ">SKYLOG ON\n")
#    lines.append(tMax + ">SNAPLOG OFF\n")
    lines.append(tMax + ">CFLLOG OFF\n")
#    lines.append(tMax + ">EVTLOG OFF\n")
    lines.append(tMax + ">SKYLOG OFF\n")

    #%% Step 5: Write 'lines' list to Scenario file
    
    # open the file
    scenName    = "settings_%s_%i-%i.scn" %(method, int(vrange[0]), int(vrange[1]))    
    directory   = settingsFilesDir
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
