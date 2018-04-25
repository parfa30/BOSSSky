#!/usr/bin/env python
"""
Title: Moon Model
Author: P. Fagrelius

Based on Jones et al 2013 to get Istar and then Noll et al 2012 for scattering 

"""

import numpy as np 
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
from astropy.io import fits
from scipy import interpolate
import astropy.table
import speclite.filters
import astropy.units as u 

sys.path.insert(0,'/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/python/')
from sky_model import Sky

class MoonModel(Sky):
    def __init__(self, ObsMeta, wave_range=None):
        Sky.__init__(self, ObsMeta, wave_range=wave_range)
        self.name = 'Moon'

        #Get V-band filter
        self.vfilter = speclite.filters.load_filter('bessell-V')
      
        self.SA_M = 6.4177*10**(-5) #sr solid angle of moon
        self.arcsec2 = np.pi #size of fiber

        self.wave_range_A = self.wave_range*10

        self.ext_zenith = self.extinction_curve(1)
        self.ext_curve = self.extinction_curve(self.ObsMeta['AIRMASS'])
        self.moon_ext_curve = self.extinction_curve(self.ObsMeta['MOON_X'])
        self.obs_ext_curve = self.extinction_curve(self.ObsMeta['OBS_X'])

        self.a1 = 12.73
        self.a2 = 0.025
        self.a3 = 4
        self.a4 = 16.57
        self.a5 = 5.36
        self.a6 = 1.06
        self.a7 = 6.15
        self.a8 = 40

        #print("Moon Model Initialized")


    def calc_albedo(self):
        #Setup albedo constants from Kieffer & Stone 2011
        albedo_file = '/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/albedo_constants.csv'
        albedo_table = pd.read_csv(albedo_file, delim_whitespace=True)
        AlbedoConstants = {}
        for constant in list(albedo_table):
            line = interpolate.interp1d(albedo_table['WAVELENGTH'],albedo_table[constant],bounds_error=False, fill_value=0)
            AlbedoConstants[constant] = line 

        p1 = 4.06054
        p2 = 12.8802
        p3 = -30.5858
        p4 = 16.7498
        #IS LUNAR PHASE BEING TREATED CORRECTLY HERE?

        A = []
        for i in range(4):
            A.append(AlbedoConstants['a%s'%str(i)](self.wave_range_A)*(self.ObsMeta['MOON_PHASE']**i))
        for j in range(1,4):
            A.append(AlbedoConstants['b%s'%str(j)](self.wave_range_A)*(self.ObsMeta['SOLAR_SELENO']**(2*j-1)))
        A.append(AlbedoConstants['d1'](self.wave_range_A)*np.exp(-self.ObsMeta['MOON_PHASE']/p1))
        A.append(AlbedoConstants['d2'](self.wave_range_A)*np.exp(-self.ObsMeta['MOON_PHASE']/p2))
        A.append(AlbedoConstants['d3'](self.wave_range_A)*np.cos(self.ObsMeta['MOON_PHASE']-p3)/p4)
        lnA = np.sum(A,axis=0)
        A = np.exp(lnA)
        self.albedo = np.array(A) 


    def get_V_ext(self):
        V = self.vfilter.get_ab_magnitude(self.moon_spectrum, self.wave_range_A)
        Vstar = self.vfilter.get_ab_magnitude(self.moon_spectrum * self.ext_zenith, self.wave_range_A)

        self.k_V = Vstar - V
    
    def get_scattered_moon_V(self):
        self.get_V_ext()

        m = -self.a1+self.a2*np.abs(self.ObsMeta['MOON_PHASE'])+self.a3*10**(-9)*self.ObsMeta['MOON_PHASE']**(4)
        I_V = 10**(-0.4*(m+self.a4))

        moon_ext = 10**(-0.4*self.k_V*self.ObsMeta['MOON_X'])*(1-10**(-0.4*self.k_V*self.ObsMeta['OBS_X']))

        fR = 10**(self.a5)*(self.a6+(np.cos(np.deg2rad(self.ObsMeta['MOON_SEP'])))**(2))
        fM = 10**(self.a7 - (self.ObsMeta['MOON_SEP']/self.a8))

        #m = -12.73+0.026*np.abs(self.ObsMeta['MOON_PHASE'])+4*10**(-9)*self.ObsMeta['MOON_PHASE']**(4)
        #I_V = 10**(-0.4*(m+16.57))

        #moon_ext = 10**(-0.4*self.k_V*self.ObsMeta['MOON_X'])*(1-10**(-0.4*self.k_V*self.ObsMeta['OBS_X']))

        #fR = 10**(5.36)*(1.06+(np.cos(np.deg2rad(self.ObsMeta['MOON_SEP'])))**(2))
        #fM = 10**(6.15 - (self.ObsMeta['MOON_SEP']/40.))

        B = (fR+fM)*I_V*moon_ext
        self.V_scatt = (20.7233 - np.log(B)/34.08)/0.92104


    def get_spectrum(self):
        self.calc_albedo()
        self.moon_spectrum = self.solar_spectrum*(self.SA_M/np.pi)*(384400/self.ObsMeta['MOON_D'])**2 * self.albedo

        self.get_scattered_moon_V()

        self.I = self.moon_spectrum * self.moon_ext_curve * (1 - self.obs_ext_curve)
        self.V = self.vfilter.get_ab_magnitude(self.I, self.wave_range_A)

        #Scale by the Scattered V brightness
        spectrum = self.I * 10 ** (-0.4*(self.V_scatt - self.V)) * self.arcsec2*10**(17) #10^-17 erg/cm2/s/A
        spectrum[np.isnan(spectrum)] = 0
        self.flux = spectrum

    def test(self): 
        self.get_spectrum()
        plt.plot(self.wave_range, self.flux, label = 'Scattered moonlight')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux (10^-17 erg/cm2/s/A)")


if __name__ == '__main__':
    obs_meta = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/test_ObsMeta.npy')
    M = MoonSpectrum(obs_meta)
    M.test()
    plt.show()
    




