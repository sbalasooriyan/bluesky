'''
logHeader.py

Contains functions to write the header for the different log files

'''


def snapHeader():
  parameters = "SimTime [s], "  + \
               "Aircraft ID [-], " + \
               "Latitude [deg], " + \
               "Longitude [deg], " + \
               "Altitude [m], " + \
               "Heading [deg], " + \
               "TAS [m/s], " + \
               "GS [m/s], " + \
               "CAS [m/s], " + \
               "Mach [-], " + \
               "Pilot ALT [m], " + \
               "Pilot SPD (CAS) [m/s], " + \
               "Pilot SPD (TAS) [m/s], " + \
               "Pilot HDG [deg]," + \
               "Distance [m], " + \
               "Work [J]"

  lines      = "#######################################################\n" + \
               "SNAP LOG\n" + \
               "Airspace Snapshot Data\n" + \
               "DT: %s [s]\n" + \
               "#######################################################\n\n" + \
               "Parameters [Units]:\n" + \
               parameters + "\n"

  return lines

def cflHeader():
  parameters = "SimTime [s], "  + \
               "Aircraft ID [-], " + \
               "Latitude [deg], " + \
               "Longitude [deg], " + \
               "Altitude [m], " + \
               "Heading [deg], " + \
               "TAS [m/s], " + \
               "GS [m/s], " + \
               "CAS [m/s], " + \
               "Mach [-], " + \
               "Pilot ALT [m], " + \
               "Pilot SPD (CAS) [m/s], " + \
               "Pilot SPD (TAS) [m/s], " + \
               "Pilot HDG [deg]," + \
               "Distance [m], " + \
               "Work [J]"

  lines      = "#######################################################\n" + \
               "CFL LOG\n" + \
               "Conflict Data\n" + \
               "DT: %s [s]\n" + \
               "#######################################################\n\n" + \
               "Parameters [Units]:\n" + \
               parameters + "\n"

  return lines

def skyHeader():
  parameters = "SimTime [s], "  + \
               "Aircraft ID [-], " + \
               "Latitude [deg], " + \
               "Longitude [deg], " + \
               "Altitude [m], " + \
               "Heading [deg], " + \
               "TAS [m/s], " + \
               "GS [m/s], " + \
               "CAS [m/s], " + \
               "Mach [-], " + \
               "Pilot ALT [m], " + \
               "Pilot SPD (CAS) [m/s], " + \
               "Pilot SPD (TAS) [m/s], " + \
               "Pilot HDG [deg], " + \
               "Distance [m], " + \
               "Work [J], " + \
               "StartTime [s], " + \
               "Severity [-], " + \
               "Ntraf SIM [-], " + \
               "Ntraf EXPT [-]"

  lines      = "#######################################################\n" + \
               "SKY LOG\n" + \
               "Event Data\n" + \
               "DT: %s [s]\n" + \
               "#######################################################\n\n" + \
               "Parameters [Units]:\n" + \
               parameters + "\n"

  return lines

def evtHeader():
  parameters = "SimTime [s], "  + \
               "Event Description [-]"

  lines      = "#######################################################\n" + \
               "EVT LOG\n" + \
               "Events\n" + \
               "DT: %s [s]\n" + \
               "#######################################################\n\n" + \
               "Parameters [Units]:\n" + \
               parameters + "\n"

  return lines