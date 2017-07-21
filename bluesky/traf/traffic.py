import numpy as np
from math import *
from random import random, randint
from ..tools import datalog, geo, logHeader
from ..tools.misc import latlon2txt
from ..tools.aero import fpm, kts, ft, g0, Rearth, nm, \
                         vatmos,  vtas2cas, vtas2mach, casormach, vcasormach

from ..tools.dynamicarrays import DynamicArrays, RegisterElementParameters

from windsim import WindSim

from trails import Trails
from adsbmodel import ADSB
from asas import ASAS
from pilot import Pilot
from autopilot import Autopilot
from activewpdata import ActiveWaypoint
from turbulence import Turbulence
from area import Area

from .. import settings

try:
    if settings.performance_model == 'bluesky':
        print 'Using BlueSky performance model'
        from perf import Perf

    elif settings.performance_model == 'bada':
        from perfbada import PerfBADA as Perf

except ImportError as err:
    print err.args[0]
    print 'Falling back to BlueSky performance model'
    from perf import Perf


class Traffic(DynamicArrays):
    """
    Traffic class definition    : Traffic data
    Methods:
        Traffic()            :  constructor
        reset()              :  Reset traffic database w.r.t a/c data
        create(acid,actype,aclat,aclon,achdg,acalt,acspd) : create aircraft
        delete(acid)         : delete an aircraft from traffic data
        deletall()           : delete all traffic
        update(sim)          : do a numerical integration step
        id2idx(name)         : return index in traffic database of given call sign
        engchange(i,engtype) : change engine type of an aircraft
        setNoise(A)          : Add turbulence
    Members: see create
    Created by  : Jacco M. Hoekstra
    """

    def __init__(self, navdb):
        self.wind = WindSim()

        # Define the periodic loggers
        datalog.definePeriodicLogger('SNAPLOG', logHeader.snapHeader(), 10.0)
        datalog.definePeriodicLogger('CFLLOG' , logHeader.cflHeader() ,  1.0)
        # Define event-based loggers
        self.skylog = datalog.defineLogger('SKYLOG', logHeader.skyHeader())
        self.evtlog = datalog.defineLogger('EVTLOG', logHeader.evtHeader())

        with RegisterElementParameters(self):

            # Register the following parameters for logging
            with datalog.registerLogParameters('SNAPLOG', self):
                # Aircraft Info
                self.id      = []  # identifier (string)

                # Positions
                self.lat        = np.array([])  # latitude [deg]
                self.lon        = np.array([])  # longitude [deg]
                self.alt        = np.array([])  # altitude [m]
                self.hdg        = np.array([])  # traffic heading [deg]

                # Velocities
                self.tas        = np.array([])  # true airspeed [m/s]
                self.gs         = np.array([])  # ground speed [m/s]
                self.cas        = np.array([])  # calibrated airspeed [m/s]
                self.M          = np.array([])  # mach number

                # Traffic autopilot settings
                self.apalt      = np.array([])  # selected alt[m]
                self.aspd       = np.array([])  # selected spd(CAS) [m/s]
                self.aptas      = np.array([])  # just for initializing
                self.aphdg      = np.array([])  # selected heading [deg]

                # Efficiency related variables
                self.dist       = np.array([])   # Horizontal flight distance [m]
                self.work       = np.array([])   # Work Done [J]
                

                
            

            # Originally in snaplog
            self.type    = []  # aircaft type (string)
            self.trk     = np.array([])  # track angle [deg]
            self.gsnorth = np.array([])  # ground speed [m/s]
            self.gseast  = np.array([])  # ground speed [m/s]
            self.vs      = np.array([])  # vertical speed [m/s]
            self.p       = np.array([])  # air pressure [N/m2]
            self.rho     = np.array([])  # air density [kg/m3]
            self.Temp    = np.array([])  # air temperature [K]
            self.dtemp   = np.array([])  # delta t for non-ISA conditions
            self.ama    = np.array([])  # selected spd above crossover altitude (Mach) [-]
            self.avs    = np.array([])  # selected vertical speed [m/s]

            # Whether to perform LNAV and VNAV
            self.swlnav   = np.array([], dtype=np.bool)
            self.swvnav   = np.array([], dtype=np.bool)

            # Flight Models
            self.asas   = ASAS(self)
            self.ap     = Autopilot(self)
            self.pilot  = Pilot(self)
            self.adsb   = ADSB(self)
            self.trails = Trails(self)
            self.actwp  = ActiveWaypoint(self)

            # Traffic performance data
            self.avsdef = np.array([])  # [m/s]default vertical speed of autopilot
            self.aphi   = np.array([])  # [rad] bank angle setting of autopilot
            self.ax     = np.array([])  # [m/s2] absolute value of longitudinal accelleration
            self.bank   = np.array([])  # nominal bank angle, [radian]
            self.hdgsel = np.array([], dtype=np.bool)  # determines whether aircraft is turning

            # Crossover altitude
            self.abco   = np.array([])
            self.belco  = np.array([])

            # limit settings
            self.limspd      = np.array([])  # limit speed
            self.limspd_flag = np.array([], dtype=np.bool)  # flag for limit spd - we have to test for max and min
            self.limalt      = np.array([])  # limit altitude
            self.limvs       = np.array([])  # limit vertical speed due to thrust limitation
            self.limvs_flag  = np.array([])

            # Display information on label
            self.label       = []  # Text and bitmap of traffic label

            # Miscallaneous
            self.coslat = np.array([])  # Cosine of latitude for computations
            self.eps    = np.array([])  # Small nonzero numbers

        
        with datalog.registerLogParameters('CFLLOG', self):
            # Aircraft Info
            self.cflid      = []  # identifier (string)

            # Positions
            self.cfllat     = np.array([])  # latitude [deg]
            self.cfllon     = np.array([])  # longitude [deg]
            self.cflalt     = np.array([])  # altitude [m]
            self.cflhdg     = np.array([])  # traffic heading [deg]

            # Velocities
            self.cfltas     = np.array([])  # true airspeed [m/s]
            self.cflgs      = np.array([])  # ground speed [m/s]
            self.cflcas     = np.array([])  # calibrated airspeed [m/s]
            self.cflM       = np.array([])  # mach number

            # Traffic autopilot settings
            self.cflapalt   = np.array([])  # selected alt[m]
            self.cflaspd    = np.array([])  # selected spd(CAS) [m/s]
            self.cflaptas   = np.array([])  # just for initializing
            self.cflaphdg   = np.array([])  # selected heading [deg]
            
            # Traffic ASAS settings
            self.cflasasspd = np.array([])  # resolution speed [m/s]
            self.cflasashdg = np.array([])  # resolution heading [deg]

            # Efficiency related variables
            self.cfldist    = np.array([])   # Horizontal flight distance [m]
            self.cflwork    = np.array([])   # Work Done [J]
            
        with datalog.registerLogParameters('SKYLOG', self):
            # Aircraft Info
            self.skyid      = []  # identifier (string)
            # Event Info
            self.skyevt     = []  # short description (string)

            # Positions
            self.skylat     = np.array([])  # latitude [deg]
            self.skylon     = np.array([])  # longitude [deg]
            self.skyalt     = np.array([])  # altitude [m]
            self.skyhdg     = np.array([])  # traffic heading [deg]

            # Velocities
            self.skytas     = np.array([])  # true airspeed [m/s]
            self.skygs      = np.array([])  # ground speed [m/s]
            self.skycas     = np.array([])  # calibrated airspeed [m/s]
            self.skyM       = np.array([])  # mach number

            # Traffic autopilot settings
            self.skyapalt   = np.array([])  # selected alt[m]
            self.skyaspd    = np.array([])  # selected spd(CAS) [m/s]
            self.skyaptas   = np.array([])  # just for initializing
            self.skyaphdg   = np.array([])  # selected heading [deg]
            
            # Traffic ASAS settings
            self.skyasasspd = np.array([])  # resolution speed [m/s]
            self.skyasashdg = np.array([])  # resolution heading [deg]

            # Efficiency related variables
            self.skydist    = np.array([])   # Horizontal flight distance [m]
            self.skywork    = np.array([])   # Work Done [J]
            
        with datalog.registerLogParameters('EVTLOG', self):
            # Event Info
            self.evtstr      = []  # Description of event
        
        
        
        # Default bank angles per flight phase
        self.bphase = np.deg2rad(np.array([15, 35, 35, 35, 15, 45]))
        # Logger vars
        self.evtcfl = []
        self.evtlos = []
        
        self.reset(navdb)

    def reset(self, navdb):
        # This ensures that the traffic arrays (which size is dynamic)
        # are all reset as well, so all lat,lon,sdp etc but also objects adsb
        super(Traffic, self).reset()
        self.ntraf = 0

        # Reset models
        self.wind.clear()

        # Build new modules for area and turbulence
        self.area       = Area(self)
        self.Turbulence = Turbulence(self)

        # Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)
        self.setNoise(False)

        # Import navigation data base
        self.navdb   = navdb

        # Default: BlueSky internal performance model.
        # Insert your BADA files to the folder "BlueSky/data/coefficients/BADA"
        # for working with EUROCONTROL`s Base of Aircraft Data revision 3.12
        self.perf    = Perf(self)
        self.trails.reset()

    def mcreate(self, count, actype=None, alt=None, spd=None, dest=None, area=None):
        """ Create multiple random aircraft in a specified area """
        idbase = chr(randint(65, 90)) + chr(randint(65, 90))
        if actype is None:
            actype = 'B744'

        n = count
        super(Traffic, self).create(n)

        # Increase number of aircraft
        self.ntraf = self.ntraf + count

        acids = []
        aclats = []
        aclons = []
        achdgs = []
        acalts = []
        acspds = []

        for i in xrange(count):
            acids.append((idbase + '%05d' % i).upper())
            aclats.append(random() * (area[1] - area[0]) + area[0])
            aclons.append(random() * (area[3] - area[2]) + area[2])
            achdgs.append(float(randint(1, 360)))
            acalts.append((randint(2000, 39000) * ft) if alt is None else alt)
            acspds.append((randint(250, 450) * kts) if spd is None else spd)

        # Aircraft Info
        self.id[-n:]   = acids
        self.type[-n:] = [actype] * n

        # Positions
        self.lat[-n:]  = aclats
        self.lon[-n:]  = aclons
        self.alt[-n:]  = acalts

        self.hdg[-n:]  = achdgs
        self.trk[-n:]  = achdgs

        # Velocities
        self.tas[-n:], self.cas[-n:], self.M[-n:] = vcasormach(acspds, acalts)
        self.gs[-n:]      = self.tas[-n:]
        self.gsnorth[-n:] = self.tas[-n:] * np.cos(np.radians(self.hdg[-n:]))
        self.gseast[-n:]  = self.tas[-n:] * np.sin(np.radians(self.hdg[-n:]))

        # Atmosphere
        self.p[-n:], self.rho[-n:], self.Temp[-n:] = vatmos(acalts)

        # Wind
        if self.wind.winddim > 0:
            vnwnd, vewnd     = self.wind.getdata(self.lat[-n:], self.lon[-n:], self.alt[-n:])
            self.gsnorth[-n:] = self.gsnorth[-n:] + vnwnd
            self.gseast[-n:]  = self.gseast[-n:]  + vewnd
            self.trk[-n:]     = np.degrees(np.arctan2(self.gseast[-n:], self.gsnorth[-n:]))
            self.gs[-n:]      = np.sqrt(self.gsnorth[-n:]**2 + self.gseast[-n:]**2)

        # Traffic performance data
        #(temporarily default values)
        self.avsdef[-n:] = 1500. * fpm   # default vertical speed of autopilot
        self.aphi[-n:]   = np.radians(25.)  # bank angle setting of autopilot
        self.ax[-n:]     = kts           # absolute value of longitudinal accelleration
        self.bank[-n:]   = np.radians(25.)

        # Crossover altitude
        self.abco[-n:]   = 0  # not necessary to overwrite 0 to 0, but leave for clarity
        self.belco[-n:]  = 1

        # Traffic autopilot settings
        self.aspd[-n:]  = self.cas[-n:]
        self.aptas[-n:] = self.tas[-n:]
        self.aphdg[-n:] = self.hdg[-n:]
        self.apalt[-n:] = self.alt[-n:]

        # Display information on label
        self.label[-n:] = ['', '', '', 0]

        # Miscallaneous
        self.coslat[-n:] = np.cos(np.radians(aclats))  # Cosine of latitude for flat-earth aproximations
        self.eps[-n:] = 0.01
        
        # Efficiency related variables
        # not necessary to overwrite 0 to 0, but leave for clarity
        self.dist[-n:] = 0.0   # Horizontal flight distance [m]
        self.work[-n:] = 0.0   # Work Done [J]

        # ----- Submodules of Traffic -----
        self.ap.create(n)
        self.actwp.create(n)
        self.pilot.create(n)
        self.adsb.create(n)
        self.area.create(n)
        self.asas.create(n)
        self.perf.create(n)
        self.trails.create(n)


    def create(self, acid=None, actype="B744", aclat=None, aclon=None, achdg=None, acalt=None, casmach=None):
        """Create an aircraft"""

        # Check if not already exist
        if self.id.count(acid.upper()) > 0:
            return False, acid + " already exists."  # already exists do nothing

        # Catch missing acid, replace by a default
        if acid is None or acid == "*":
            acid = "KL204"
            flno = 204
            while self.id.count(acid) > 0:
                flno = flno + 1
                acid = "KL" + str(flno)

        # Check for (other) missing arguments
        if actype is None or aclat is None or aclon is None or achdg is None \
                or acalt is None or casmach is None:

            return False, "CRE: Missing one or more arguments:"\
                          "acid,actype,aclat,aclon,achdg,acalt,acspd"

        super(Traffic, self).create()

        # Increase number of aircraft
        self.ntraf = self.ntraf + 1

        # Aircraft Info
        self.id[-1]   = acid.upper()
        self.type[-1] = actype

        # Positions
        self.lat[-1]  = aclat
        self.lon[-1]  = aclon
        self.alt[-1]  = acalt

        self.hdg[-1]  = achdg
        self.trk[-1]  = achdg

        # Velocities
        self.tas[-1], self.cas[-1], self.M[-1] = casormach(casmach, acalt)
        self.gs[-1]      = self.tas[-1]
        self.gsnorth[-1] = self.tas[-1] * cos(radians(self.hdg[-1]))
        self.gseast[-1]  = self.tas[-1] * sin(radians(self.hdg[-1]))

        # Atmosphere
        self.p[-1], self.rho[-1], self.Temp[-1] = vatmos(acalt)

        # Wind
        if self.wind.winddim > 0:
            vnwnd, vewnd     = self.wind.getdata(self.lat[-1], self.lon[-1], self.alt[-1])
            self.gsnorth[-1] = self.gsnorth[-1] + vnwnd
            self.gseast[-1]  = self.gseast[-1]  + vewnd
            self.trk[-1]     = np.degrees(np.arctan2(self.gseast[-1], self.gsnorth[-1]))
            self.gs[-1]      = np.sqrt(self.gsnorth[-1]**2 + self.gseast[-1]**2)

        # Traffic performance data
        #(temporarily default values)
        self.avsdef[-1] = 1500. * fpm   # default vertical speed of autopilot
        self.aphi[-1]   = radians(25.)  # bank angle setting of autopilot
        self.ax[-1]     = kts           # absolute value of longitudinal accelleration
        self.bank[-1]   = radians(25.)

        # Crossover altitude
        self.abco[-1]   = 0  # not necessary to overwrite 0 to 0, but leave for clarity
        self.belco[-1]  = 1

        # Traffic autopilot settings
        self.aspd[-1]  = self.cas[-1]
        self.aptas[-1] = self.tas[-1]
        self.aphdg[-1] = self.hdg[-1]
        self.apalt[-1] = self.alt[-1]

        # Display information on label
        self.label[-1] = ['', '', '', 0]
        
        # Efficiency related variables
        self.dist[-1] = 0.0   # Horizontal flight distance [m]
        self.work[-1] = 0.0   # Work Done [J]

        # Miscallaneous
        self.coslat[-1] = cos(radians(aclat))  # Cosine of latitude for flat-earth aproximations
        self.eps[-1] = 0.01

        # ----- Submodules of Traffic -----
        self.ap.create()
        self.actwp.create()
        self.pilot.create()
        self.adsb.create()
        self.area.create()
        self.asas.create()
        self.perf.create()
        self.trails.create()
        
        # Logger
        self.UpdateEvtLog('cre', -1)

        return True

    def creconfs(self, acid, actype, targetidx, dpsi, cpa, tlosh, dH=None, tlosv=None, spd=None):
        latref  = self.lat[targetidx]  # deg
        lonref  = self.lon[targetidx]  # deg
        altref  = self.alt[targetidx]  # m
        trkref  = radians(self.trk[targetidx])
        gsref   = self.gs[targetidx]   # m/s
        vsref   = self.vs[targetidx]   # m/s
        cpa     = cpa * nm
        pzr     = settings.asas_pzr * nm
        pzh     = settings.asas_pzh * ft

        trk     = trkref + radians(dpsi)
        gs      = gsref if spd is None else spd
        if dH is None:
            acalt = altref
            acvs  = 0.0
        else:
            acalt = altref + dH
            tlosv = tlosh if tlosv is None else tlosv
            acvs  = vsref - np.sign(dH) * (abs(dH) - pzh) / tlosv

        # Horizontal relative velocity vector
        gsn, gse     = gs    * cos(trk),          gs    * sin(trk)
        vreln, vrele = gsref * cos(trkref) - gsn, gsref * sin(trkref) - gse
        # Relative velocity magnitude
        vrel    = sqrt(vreln * vreln + vrele * vrele)
        # Relative travel distance to closest point of approach
        drelcpa = tlosh * vrel + sqrt(pzr * pzr - cpa * cpa)
        # Initial intruder distance
        dist    = sqrt(drelcpa * drelcpa + cpa * cpa)
        # Rotation matrix diagonal and cross elements for distance vector
        rd      = drelcpa / dist
        rx      = cpa / dist
        # Rotate relative velocity vector to obtain intruder bearing
        brn     = degrees(atan2(-rx * vreln + rd * vrele,
                                 rd * vreln + rx * vrele))

        # Calculate intruder lat/lon
        aclat, aclon = geo.qdrpos(latref, lonref, brn, dist / nm)

        # convert groundspeed to CAS, and track to heading
        wn, we     = self.wind.getdata(aclat, aclon, acalt)
        tasn, tase = gsn - wn, gse - we
        acspd      = vtas2cas(sqrt(tasn * tasn + tase * tase), acalt)
        achdg      = degrees(atan2(tase, tasn))

        # Create and, when necessary, set vertical speed
        self.create(acid, actype, aclat, aclon, achdg, acalt, acspd)
        self.ap.selalt(len(self.lat) - 1, altref, acvs)
        self.vs[-1] = acvs

    def delete(self, acid):
        """Delete an aircraft"""

        # Look up index of aircraft
        idx = self.id2idx(acid)
        # Do nothing if not found
        if idx < 0:
            return False
        # Decrease number of aircraft
        self.ntraf = self.ntraf - 1
        
        # Logger (call it before actually deleting!!)
        self.UpdateEvtLog('del', idx)

        # Delete all aircraft parameters
        super(Traffic, self).delete(idx)

        # ----- Submodules of Traffic -----
        self.perf.delete(idx)
        self.area.delete(idx)
        return True

    def update(self, simt, simdt):
        # Update only if there is traffic ---------------------
        if self.ntraf == 0:
            return

        #---------- Atmosphere --------------------------------
        self.p, self.rho, self.Temp = vatmos(self.alt)

        #---------- ADSB Update -------------------------------
        self.adsb.update(simt)

        #---------- Fly the Aircraft --------------------------
        self.ap.update(simt)
        self.asas.update(simt)
        self.pilot.FMSOrAsas()

        #---------- Limit Speeds ------------------------------
        self.pilot.FlightEnvelope()

        #---------- Kinematics --------------------------------
        self.UpdateAirSpeed(simdt, simt)
        self.UpdateGroundSpeed(simdt)
        self.UpdatePosition(simdt)

        #---------- Performance Update ------------------------
        self.perf.perf(simt)

        #---------- Simulate Turbulence -----------------------
        self.Turbulence.Woosh(simdt)

        #---------- Aftermath ---------------------------------
        self.trails.update(simt)
        self.area.check(simt)
        
        #---------- Flight Efficiency Update ------------------
        self.UpdateEfficiency(simdt)
        
        #---------- Loggers -----------------------------------
        self.UpdateTrafCflLog()
        self.UpdateEvtLog('updateconf')
        
        return

    def UpdateAirSpeed(self, simdt, simt):
        # Acceleration
        self.delspd = self.pilot.spd - self.tas
        
        swspdsel = np.abs(self.delspd) > 0.4  # <1 kts = 0.514444 m/s
        ax = self.perf.acceleration(simdt)

        # Update velocities
        self.tas = self.tas + swspdsel * ax * np.sign(self.delspd) * simdt
        
        self.cas = vtas2cas(self.tas, self.alt)
        self.M   = vtas2mach(self.tas, self.alt)

        # Turning
        turnrate = np.degrees(g0 * np.tan(self.bank) / np.maximum(self.tas, self.eps))
        delhdg   = (self.pilot.hdg - self.hdg + 180.) % 360 - 180.  # [deg]
        self.hdgsel = np.abs(delhdg) > np.abs(2. * simdt * turnrate)

        # Update heading
        self.hdg = (self.hdg + simdt * turnrate * self.hdgsel * np.sign(delhdg)) % 360.

        # Update vertical speed
        delalt   = self.pilot.alt - self.alt
        self.swaltsel = np.abs(delalt) > np.maximum(10 * ft, np.abs(2. * simdt * np.abs(self.vs)))
        self.vs  = self.swaltsel * np.sign(delalt) * self.pilot.vs

    def UpdateGroundSpeed(self, simdt):
        # Compute ground speed and track from heading, airspeed and wind
        if self.wind.winddim == 0:  # no wind
            self.gsnorth  = self.tas * np.cos(np.radians(self.hdg))
            self.gseast   = self.tas * np.sin(np.radians(self.hdg))

            self.gs  = self.tas
            self.trk = self.hdg

        else:
            windnorth, windeast = self.wind.getdata(self.lat, self.lon, self.alt)
            self.gsnorth  = self.tas * np.cos(np.radians(self.hdg)) + windnorth
            self.gseast   = self.tas * np.sin(np.radians(self.hdg)) + windeast

            self.gs  = np.sqrt(self.gsnorth**2 + self.gseast**2)
            self.trk = np.degrees(np.arctan2(self.gseast, self.gsnorth)) % 360.

    def UpdatePosition(self, simdt):
        # Update position
        self.alt = np.where(self.swaltsel, self.alt + self.vs * simdt, self.pilot.alt)
        self.lat = self.lat + np.degrees(simdt * self.gsnorth / Rearth)
        self.coslat = np.cos(np.deg2rad(self.lat))
        self.lon = self.lon + np.degrees(simdt * self.gseast / self.coslat / Rearth)

    def id2idx(self, acid):
        """Find index of aircraft id"""
        try:
            return self.id.index(acid.upper())
        except:
            return -1

    def setNoise(self, noise=None):
        """Noise (turbulence, ADBS-transmission noise, ADSB-truncated effect)"""
        if noise is None:
            return True, "Noise is currently " + ("on" if self.Turbulence.active else "off")

        self.Turbulence.SetNoise(noise)
        self.adsb.SetNoise(noise)
        return True

    def engchange(self, acid, engid):
        """Change of engines"""
        self.perf.engchange(acid, engid)
        return

    def move(self, idx, lat, lon, alt=None, hdg=None, casmach=None, vspd=None):
        self.lat[idx]      = lat
        self.lon[idx]      = lon

        if alt:
            self.alt[idx]   = alt
            self.apalt[idx] = alt

        if hdg:
            self.hdg[idx]  = hdg
            self.ap.trk[idx] = hdg

        if casmach:
            self.tas[idx], self.aspd[-1], dummy = casormach(casmach, alt)

        if vspd:
            self.vs[idx]       = vspd
            self.swvnav[idx] = False

    def nom(self, idx):
        """ Reset acceleration back to nominal (1 kt/s^2): NOM acid """
        self.ax[idx] = kts

    def poscommand(self, scr, idxorwp):# Show info on aircraft(int) or waypoint or airport (str)
        """POS command: Show info or an aircraft, airport, waypoint or navaid"""
        # Aircraft index
        if type(idxorwp)==int and idxorwp >= 0:

            idx           = idxorwp
            acid          = self.id[idx]
            actype        = self.type[idx]
            latlon        = latlon2txt(self.lat[idx], self.lon[idx])
            alt           = round(self.alt[idx] / ft)
            hdg           = round(self.hdg[idx])
            trk           = round(self.trk[idx])
            cas           = round(self.cas[idx] / kts)
            tas           = round(self.tas[idx] / kts)
            gs            = round(self.gs[idx]/kts)
            M             = self.M[idx]
            VS            = round(self.vs[idx]/ft*60.)
            route         = self.ap.route[idx]

            # Position report

            lines = "Info on %s %s index = %d\n" %(acid, actype, idx)     \
                  + "Pos: "+latlon+ "\n"                                  \
                  + "Hdg: %03d   Trk: %03d\n"        %(hdg, trk)              \
                  + "Alt: %d ft  V/S: %d fpm\n"  %(alt,VS)                \
                  + "CAS/TAS/GS: %d/%d/%d kts   M: %.3f\n"%(cas,tas,gs,M)

            # FMS AP modes
            if self.swlnav[idx] and route.nwp > 0 and route.iactwp >= 0:

                if self.swvnav[idx]:
                    lines = lines + "VNAV, "

                lines += "LNAV to " + route.wpname[route.iactwp] + "\n"

            # Flight info: Destination and origin
            if self.ap.orig[idx] != "" or self.ap.dest[idx] != "":
                lines = lines +  "Flying"

                if self.ap.orig[idx] != "":
                    lines = lines +  " from " + self.ap.orig[idx]

                if self.ap.dest[idx] != "":
                    lines = lines +  " to " + self.ap.dest[idx]

            # Show a/c info and highlight route of aircraft in radar window
            # and pan to a/c (to show route)
            return scr.showacinfo(acid,lines)

        # Waypoint: airport, navaid or fix
        else:
            wp = idxorwp.upper()

            # Reference position for finding nearest
            reflat = scr.ctrlat
            reflon = scr.ctrlon

            lines = "Info on "+wp+":\n"

            # First try airports (most used and shorter, hence faster list)
            iap = self.navdb.getaptidx(wp)
            if iap>=0:
                aptypes = ["large","medium","small"]
                lines = lines + self.navdb.aptname[iap]+"\n"                 \
                        + "is a "+ aptypes[max(-1,self.navdb.aptype[iap]-1)] \
                        +" airport at:\n"                                    \
                        + latlon2txt(self.navdb.aptlat[iap],                 \
                                     self.navdb.aptlon[iap]) + "\n"          \
                        + "Elevation: "                                      \
                        + str(int(round(self.navdb.aptelev[iap]/ft)))        \
                        + " ft \n"

               # Show country name
                try:
                     ico = self.navdb.cocode2.index(self.navdb.aptco[iap].upper())
                     lines = lines + "in "+self.navdb.coname[ico]+" ("+      \
                             self.navdb.aptco[iap]+")"
                except:
                     ico = -1
                     lines = lines + "Country code: "+self.navdb.aptco[iap]
                try:
                    rwytxt = str(self.navdb.rwythresholds[self.navdb.aptid[iap]].keys())
                    lines = lines + "\nRunways: " +rwytxt.strip("[]").replace("'","")
                except:
                    pass

            # Not found as airport, try waypoints & navaids
            else:
                iwps = self.navdb.getwpindices(wp,reflat,reflon)
                if iwps[0]>=0:
                    typetxt = ""
                    desctxt = ""
                    lastdesc = "XXXXXXXX"
                    for i in iwps:

                        # One line type text
                        if typetxt == "":
                            typetxt = typetxt+self.navdb.wptype[i]
                        else:
                            typetxt = typetxt+" and "+self.navdb.wptype[i]

                        # Description: multi-line
                        samedesc = self.navdb.wpdesc[i]==lastdesc
                        if desctxt == "":
                            desctxt = desctxt +self.navdb.wpdesc[i]
                            lastdesc = self.navdb.wpdesc[i]
                        elif not samedesc:
                            desctxt = desctxt +"\n"+self.navdb.wpdesc[i]
                            lastdesc = self.navdb.wpdesc[i]

                        # Navaid: frequency
                        if self.navdb.wptype[i] in ["VOR","DME","TACAN"] and not samedesc:
                            desctxt = desctxt + " "+ str(self.navdb.wpfreq[i])+" MHz"
                        elif self.navdb.wptype[i]=="NDB" and not samedesc:
                            desctxt = desctxt+ " " + str(self.navdb.wpfreq[i])+" kHz"

                    iwp = iwps[0]

                    # Basic info
                    lines = lines + wp +" is a "+ typetxt       \
                           + " at\n"\
                           + latlon2txt(self.navdb.wplat[iwp],  \
                                        self.navdb.wplon[iwp])
                    # Navaids have description
                    if len(desctxt)>0:
                        lines = lines+ "\n" + desctxt

                    # VOR give variation
                    if self.navdb.wptype[iwp]=="VOR":
                        lines = lines + "\nVariation: "+ \
                                     str(self.navdb.wpvar[iwp])+" deg"


                    # How many others?
                    nother = self.navdb.wpid.count(wp)-len(iwps)
                    if nother>0:
                        verb = ["is ","are "][min(1,max(0,nother-1))]
                        lines = lines +"\nThere "+verb + str(nother) +\
                                   " other waypoint(s) also named " + wp

                    # In which airways?
                    connect = self.navdb.listconnections(wp, \
                                                self.navdb.wplat[iwp],
                                                self.navdb.wplon[iwp])
                    if len(connect)>0:
                        awset = set([])
                        for c in connect:
                            awset.add(c[0])

                        lines = lines+"\nAirways: "+"-".join(awset)


               # Try airway id
                else:  # airway
                    awid = wp
                    airway = self.navdb.listairway(awid)
                    if len(airway)>0:
                        lines = ""
                        for segment in airway:
                            lines = lines+"Airway "+ awid + ": " + \
                                    " - ".join(segment)+"\n"
                        lines = lines[:-1] # cut off final newline
                    else:
                        return False,idxorwp+" not found as a/c, airport, navaid or waypoint"

            # Show what we found on airport and navaid/waypoint
            scr.echo(lines)

        return True

    def airwaycmd(self,scr,key=""):
        # Show conections of a waypoint

        reflat = scr.ctrlat
        reflon = scr.ctrlon

        if key=="":
            return False,'AIRWAY needs waypoint or airway'

        if self.navdb.awid.count(key)>0:
            return self.poscommand(scr, key.upper())
        else:
            # Find connecting airway legs
            wpid = key.upper()
            iwp = self.navdb.getwpidx(wpid,reflat,reflon)
            if iwp<0:
                return False,key," not found."

            wplat = self.navdb.wplat[iwp]
            wplon = self.navdb.wplon[iwp]
            connect = self.navdb.listconnections(key.upper(),wplat,wplon)
            if len(connect)>0:
                lines = ""
                for c in connect:
                    if len(c)>=2:
                        # Add airway, direction, waypoint
                        lines = lines+ c[0]+": to "+c[1]+"\n"
                scr.echo(lines[:-1])  # exclude final newline
            else:
                return False,"No airway legs found for ",key

    
    def UpdateEfficiency(self, simdt):
        # Update flight efficiency metrics
        ds = simdt * self.gs
        
        # Horizontal distance [m]
        self.dist = self.dist + ds
        
        # Work Done [J] = Force * distance; distance = spd*time
        self.work = self.work + (self.perf.Thr * ds)

           
    def UpdateTrafCflLog(self):
        # Bool-array with aircraft in conflict
        inconf = np.array([len(ids) > 0 for ids in self.asas.iconf])
        
        # Aircraft Info
        self.cflid      = [i for (i, v) in zip(self.id, inconf) if v]
        
        # Positions
        self.cfllat     = self.lat[inconf]
        self.cfllon     = self.lon[inconf]
        self.cflalt     = self.alt[inconf]
        self.cflhdg     = self.hdg[inconf]

        # Velocities
        self.cfltas     = self.tas[inconf]
        self.cflgs      = self.gs[inconf]
        self.cflcas     = self.cas[inconf]
        self.cflM       = self.M[inconf]

        # Traffic autopilot settings
        self.cflapalt   = self.apalt[inconf]
        self.cflaspd    = self.aspd[inconf]
        self.cflaptas   = self.aptas[inconf]
        self.cflaphdg   = self.aphdg[inconf]
        
        # Traffic ASAS settings
        self.cflasasspd = self.asas.spd[inconf]
        self.cflasashdg = self.asas.trk[inconf]

        # Efficiency related variables
        self.cfldist    = self.dist[inconf]
        self.cflwork    = self.work[inconf]
        
    def UpdateEvtLog(self, evt='', idx=None):
        # Because of RegisterLogParameters evtstr is set to ['']
        if self.evtstr == ['']:
            self.evtstr = []
        # Set call_log to False
        call_log = False

        if evt == 'cre':
            # Event description
            self.evtstr.append('Created ' + self.id[idx])
            self.UpdateSkyLog('CRE AC', idx)
            # Logger must be called
            call_log = True

        elif evt == 'del':
            # Event description
            self.evtstr.append('Deleted ' + self.id[idx])
            self.UpdateSkyLog('DEL AC', idx)
            # Logger must be called
            call_log = True

        elif evt == 'updateconf':
            # Get created and removed conflicts
            cre_cfl = [x for x in self.asas.conflist_now if x not in self.evtcfl]
            del_cfl = [x for x in self.evtcfl if x not in self.asas.conflist_now]
            # Get created and removed LoS
            cre_los = [x for x in self.asas.LOSlist_now if x not in self.evtlos]
            del_los = [x for x in self.evtlos if x not in self.asas.LOSlist_now]
            # Store conflict list
            self.evtcfl = self.asas.conflist_now
            # Store LoS list
            self.evtlos = self.asas.LOSlist_now
            # Check if created or removed
            if len(cre_cfl) > 0:
                for i in range(len(cre_cfl)):
                    ac1, ac2 = cre_cfl[i].split(' ')
                    self.evtstr.append('Conflict started between ' + ac1 + ' and ' + ac2)
                    self.UpdateSkyLog('CFL START ' + ac2, self.id2idx(ac1))
                    self.UpdateSkyLog('CFL START ' + ac1, self.id2idx(ac2))
                    # Logger must be called
                    call_log = True
            if len(cre_los) > 0:
                for i in range(len(cre_los)):
                    ac1, ac2 = cre_los[i].split(' ')
                    self.evtstr.append('LoS started between ' + ac1 + ' and ' + ac2)
                    self.UpdateSkyLog('LOS START ' + ac2, self.id2idx(ac1))
                    self.UpdateSkyLog('LOS START ' + ac1, self.id2idx(ac2))
                    # Logger must be called
                    call_log = True
            if len(del_cfl) > 0:
                for i in range(len(del_cfl)):
                    ac1, ac2 = del_cfl[i].split(' ')
                    self.evtstr.append('Conflict ended between ' + ac1 + ' and ' + ac2)
                    self.UpdateSkyLog('CFL END ' + ac2, self.id2idx(ac1))
                    self.UpdateSkyLog('CFL END ' + ac1, self.id2idx(ac2))
                    # Logger must be called
                    call_log = True
            if len(del_los) > 0:
                for i in range(len(del_los)):
                    ac1, ac2 = del_los[i].split(' ')
                    self.evtstr.append('LoS ended between ' + ac1 + ' and ' + ac2)
                    self.UpdateSkyLog('LOS END ' + ac2, self.id2idx(ac1))
                    self.UpdateSkyLog('LOS END ' + ac1, self.id2idx(ac2))
                    # Logger must be called
                    call_log = True

        
        if call_log:
            # Call the logger
            self.evtlog.log()
            # Reset output string
            self.evtstr = []
        
    def UpdateSkyLog(self, evt, idx):
        
        # Aircraft Info
        self.skyid      = [self.id[idx]]
        # Event Info
        self.skyevt     = [evt]
        
        # Positions
        self.skylat     = self.lat[[idx]]
        self.skylon     = self.lon[[idx]]
        self.skyalt     = self.alt[[idx]]
        self.skyhdg     = self.hdg[[idx]]

        # Velocities
        self.skytas     = self.tas[[idx]]
        self.skygs      = self.gs[[idx]]
        self.skycas     = self.cas[[idx]]
        self.skyM       = self.M[[idx]]

        # Traffic autopilot settings
        self.skyapalt   = self.apalt[[idx]]
        self.skyaspd    = self.aspd[[idx]]
        self.skyaptas   = self.aptas[[idx]]
        self.skyaphdg   = self.aphdg[[idx]]
        
        # Traffic ASAS settings
        self.skyasasspd = self.asas.spd[[idx]]
        self.skyasashdg = self.asas.trk[[idx]]

        # Efficiency related variables
        self.skydist    = self.dist[[idx]]
        self.skywork    = self.work[[idx]]
        
        # Call the logger
        self.skylog.log()
        
            
        
            
