#!/usr/bin/env python

"""
This program calculates the integrated starlight (ISL) based on the model 
in Noll et al. 

Title: ISL Model
Author: P. Fagrelius
Date: 4/11/17

Based on Noll et al 2012

To DO: 
-Add in Melchior flux
-ISL scattered flux
-extend ISL flux beyon 1000um to 1040um
"""
import sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd 
import astropy.table

sys.path.insert(0,'/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/python/')

from sky_model import Sky
class ISLModel(Sky):
    def __init__(self, ObsMeta, wave_range=None):
        Sky.__init__(self, ObsMeta, wave_range=wave_range)

        self.name = 'ISL'

        ##I0##
        self.pioneer = np.loadtxt('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/pioneer10_440.csv')
        self.GalLat = self.pioneer[0][1:]
        self.GalLon = self.pioneer[:,0][1:]
        self.P = np.array(self.pioneer[1:,1:])*self.S10*self.sr*10**(17)

        self.isl_map = interpolate.interp2d(self.GalLon, self.GalLat, self.P.T, kind='linear', bounds_error = False)

        self.I0 = self.isl_map(self.ObsMeta['GAL_LON'], self.ObsMeta['GAL_LAT'])

        Melchior_add = 2.42e-8*(7.38e-11)*10**(17)*(1/np.sin(np.deg2rad(np.abs(self.ObsMeta['GAL_LAT'])))) 
        self.base_flux = self.I0 + Melchior_add #10^-17 erg/cm2/s/A


        ##I_ISL##
        mattila = np.loadtxt('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/mattila.csv')

        isl_s = interpolate.interp1d(mattila[:,0]/10., mattila[:,1], bounds_error = False, fill_value="extrapolate")
        self.isl_spectrum = isl_s(self.wave_range)
        
        nm_440 = [np.abs(440. - w) for w in self.wave_range]
        idx = np.argmin(nm_440)
        self.relative_isl = self.isl_spectrum/self.isl_spectrum[idx]
        self.rel_sun_isl = self.solar_spectrum/self.solar_spectrum[idx]

        self.ext_curve = self.extinction_curve(self.ObsMeta['AIRMASS'])


    def get_spectrum(self):
        """Calculates the final isl spectrum after moving through atmosphere
        """
        
        self.isl_spectrum = self.base_flux*self.relative_isl # 10^-17 erg/cm2/s/A
        isl_spectrum = self.isl_spectrum * self.ext_curve
        isl_spectrum[np.isnan(isl_spectrum)] = 0

        self.flux = isl_spectrum


    def test(self): 
        self.get_spectrum()
        plt.plot(self.wave_range, self.flux, label = 'Scattered moonlight')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux (10^-17 erg/cm2/s/A)")



if __name__=="__main__":
    obs_meta = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/test_ObsMeta.npy')
    I = ISLSpectrum(obs_meta)
    I.test()
    plt.show()