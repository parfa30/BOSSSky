 #!/usr/bin/env python

""" 
This program calculates zodiacal spectrum from a solar spectrum and a 
lookup table from Leinert et al. 1998. THe output is a numpy file that 
contains a new lookup table with the spectra given the ecliptic latitude
and longitude of the observation. The solar spectrum is then redenned based
on Leinert (1998) and then scattering/ transmission applied based on 
Noll et al model.

I_Z = I0*f_co*f_abs*exp(-t_eff*X)

Title: Zodi Spectrum Generator
Author: P. Fagrelius
Date: 3/16/17

Based on Noll et al 2012

To Do:
-scattered zodi
-change test values

"""
import sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd 
import astropy.table
import astropy.units as u

sys.path.insert(0,'/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/python/')
from sky_model import Sky

class ZodiModel(Sky):
    def __init__(self, ObsMeta, wave_range=None, verbose = False):
        Sky.__init__(self, ObsMeta, wave_range=wave_range)

        self.name = 'Zodi'
        self.verbose = verbose

        ##I0##
        self.leinert_lookup = '/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/leinert_lookup_interp.npy'
        self.ZodiLookup = np.load(self.leinert_lookup)

        ##f_abs##
        #relative to 500nm
        nm_500 = [np.abs(500.-w) for w in self.wave_range]
        nm_id = np.argmin(nm_500)
        self.f_abs = self.solar_spectrum/self.solar_spectrum[nm_id]

        ##f_co##
        self.a_short = interpolate.interp1d([30,90],[1.2,0.9], bounds_error=False, fill_value=0)
        self.a_long = interpolate.interp1d([30,90],[0.8,0.6], bounds_error=False, fill_value=0)

        self.f_R0 = None
        self.f_R1 = None
        self.f_M0 = None
        self.f_M1 = None

        if self.verbose:
            print("Zodi Model Initialized")
            print(self.ObsMeta)


    def f_color(self):
        """Defines the reddening law for the solar spectrum. In leinert
        it is only given at solar elongation values of 30 and 40 so 
        'a_short' and 'a_long' are interpolations between them.
        """

        if self.elong <= 30:
            self.elong = 30
        elif self.elong >= 90:
            self.elong = 90

        f_co = np.zeros(len(self.wave_range))
        short = np.where((self.wave_range >= 200)&(self.wave_range <= 500))
        a_short = a = self.a_short(self.elong)
        c_short = 1.0 + a*np.log10(self.wave_range[short]/500.)
        f_co[short] = c_short

        longer = np.where((self.wave_range > 500)&(self.wave_range <= 2500))
        a_long = self.a_long(self.elong)
        c_long = 1.0 + a*np.log10(self.wave_range[longer]/500.)
        f_co[longer] = c_long

        f_co[np.where(f_co == 0)] = 1.0
        self.f_co = f_co

    def get_scatter_extinction_correction(self):
        """Calculates the extinction due to scattering. This is represented as 
        an effective transmission coefficient, accounting for ligth scattered in 
        and out of hte line of sight.
        """

        if self.logI <= 2.244:
            self.f_R0 = 1.407
            self.f_R1 = 2.692
        elif self.logI > 2.244:
            self.f_R0 = 0.527
            self.f_R1 = 0.715

        if self.logI <=2.255:
            self.f_M0 = 1.309
            self.f_M1 = 2.598
        elif self.logI > 2.255:
            self.f_M0 = 0.468
            self.f_M1 = 0.702
        
        self.f_ext=(self.f_R0*self.logI-self.f_R1)+(self.f_M0*self.logI-self.f_M1)
        

    def get_I0(self):
        lamb = np.argmin([np.abs(np.abs(self.ObsMeta['HELIO_LON'])-lon) for lon in self.ZodiLookup[0]])
        beta = np.argmin([np.abs(np.abs(self.ObsMeta['HELIO_LAT'])-lat) for lat in self.ZodiLookup[1]])
        self.base_zodi = self.ZodiLookup[2][lamb][beta] #S10

        self.IO = self.base_zodi*self.S10*(self.sr) #erg/cm2/s/A
        I = self.base_zodi*1.28 # 10^-8 W/m2/sr/um
        self.logI = np.log10(I)
        self.ObsMeta['LOGI'] = astropy.table.Column([self.logI])


    def get_zodi(self):
        """Gets the brightness (in S10) value of the zodiacal light
        given the ecliptic latitude and longitude. These are taken from the 
        table published in Leinert (1998) for 500nm. The brightness is converted 
        to flux and wavelength dependence (f_abs) comes from solar spectrum
        relative to 500nm.
        INPUTS: ecliptic lat and lon of the target
        OUTPUT: zodiacal spectrum at top of atm in 10^-8 W/m^2/um/sr
        """

        self.get_I0() #10^-9 erg/cm2/s/sr/A 
        self.elong = self.ObsMeta['SUN_SEP']
        self.f_color() #redenning of solar spectrum

        self.base_flux = self.IO*self.f_abs*self.f_co #Used for scattering
        self.zodi_flux = self.base_flux*10**(17)# 10^(-17) erg/cm2/s/A

    def get_spectrum(self):
        self.ext_curve = self.extinction_curve(self.ObsMeta['AIRMASS'])
        #Above atmosphere
        self.get_zodi() #10^-17 erg/cm2/s/A
        
        #Scattered Out
        self.ext_curve[np.isnan(self.ext_curve)] = 0
        self.Zodi_transmitted = self.zodi_flux*self.ext_curve

        #Scattered Out and In
        self.get_scatter_extinction_correction()
        #self.f_ext=(self.f_R0*self.logI-self.f_R1)+(self.f_M0*self.logI-self.f_M1)
        Zodi = self.zodi_flux*(self.ext_curve)**self.f_ext
        Zodi[np.isnan(Zodi)] = 0
        self.flux = Zodi

    def test(self):
        self.get_spectrum()

        plt.plot(self.wave_range, self.flux, label = 'Scattered Zodi')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux (10^-17 erg/cm2/s/A)")
        plt.legend()


if __name__=="__main__":
    obs_meta = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/test_ObsMeta.npy')
    Z = ZodiSpectrum(obs_meta)
    Z.test()
    plt.show()
