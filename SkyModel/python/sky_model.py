#!/usr/bin/env python

import numpy as np
import glob, sys, os 
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import astropy.table
from astropy.coordinates import SkyCoord
import astropy.units as u 

from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('serif')
font.set_size('large')


class Sky(object):
    def __init__(self, ObsMeta, wave_range, verbose=False):

        #Set wave range
        if wave_range is None:
            self.wave_range = np.linspace(360,1040,(1040-360))
        else:
            self.wave_range = wave_range

        self.ObsMeta = ObsMeta
        
        self.verbose = verbose
        #Zodiacal Inputs
        x = SkyCoord(ra = self.ObsMeta['RA']*u.deg, dec = self.ObsMeta['DEC']*u.deg, distance = 1000*u.AU)
        self.ObsMeta['HELIO_LON'] = astropy.table.Column(x.heliocentrictrueecliptic.lon.value)
        self.ObsMeta['HELIO_LAT'] = astropy.table.Column(x.heliocentrictrueecliptic.lat.value)
        #Additional parameters needed
        #self.ObsMeta['MOON_ILL'] = 0.4     
        moon_phase = np.arccos(2*self.ObsMeta['MOON_ILL']-1)/np.pi
        self.ObsMeta['MOON_PHASE'] = astropy.table.Column(moon_phase.astype(np.float32))
        self.ObsMeta['MOON_ZENITH'] = astropy.table.Column((90-self.ObsMeta['MOON_ALT']).astype(np.float32))
        self.ObsMeta['OBS_ZENITH'] = astropy.table.Column((90-self.ObsMeta['ALT']).astype(np.float32))
        Zmoon = np.deg2rad(90-self.ObsMeta['MOON_ALT'])
        Xmoon = (1-0.96*(np.sin(Zmoon))**(2))**(-0.5)
        Xobs = (1-0.96*(np.sin(self.ObsMeta['OBS_ZENITH']))**(2))**(-0.5)
        self.ObsMeta['MOON_X'] = astropy.table.Column(Xmoon.astype(np.float32))
        self.ObsMeta['OBS_X'] = astropy.table.Column(Xmoon.astype(np.float32))
        self.ObsMeta['SOLAR_SELENO'] = 0.17

        if self.ObsMeta['MOON_ALT'] >= self.ObsMeta['SUN_ALT']:
            sun_sep = self.ObsMeta['MOON_SEP'] + self.ObsMeta['SUN_MOON_SEP']
        elif self.ObsMeta['MOON_ALT'] < self.ObsMeta['SUN_ALT']:
            sun_sep = self.ObsMeta['MOON_SEP'] - self.ObsMeta['SUN_MOON_SEP']

        self.ObsMeta['SUN_SEP'] = astropy.table.Column(sun_sep.astype(np.float32))

        self.S10 = 1.28*10**(-9) #erg/cm2/s/A/sr
        #self.sr = 4.25*10**(10) #arcsec^2/sr
        self.sr = 7.38e-11
        #self.fiber_area = np.pi #arcsec^2  
   

        sun_s = np.loadtxt('/Users/parkerf/Research/SkyModel/DESI_Sky/solarspec.txt')
        sun_s = interpolate.interp1d(sun_s[:,1], sun_s[:,2], bounds_error = False, fill_value=0)
        self.solar_spectrum = sun_s(self.wave_range*10)

        # Get Zenith extinction coefficients
        ext_df = np.loadtxt('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/ZenithExtinction-KPNO.dat')
        #ext_df = np.genfromtxt('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/kpnoextinct.dat')
        ext = astropy.table.Table(ext_df, names = ('WAVE', 'EXT'))
        ext_c = interpolate.interp1d(ext['WAVE'], ext['EXT'], bounds_error=False, fill_value=0)
        self.zen_ext = ext_c(self.wave_range*10)


        # Check that have all necessary meta data to run the model
        needed_params = np.array(['MOON_PHASE','MOON_X', 'MOON_ALT',  'AIRMASS', 'ECL_LON', 'ECL_LAT', 'SUN_LON',
            'MOON_ILL','SOLAR_SELENO','MOON_D', 'MOON_ALT', 'SUN_ALT', 'SUN_MOON_SEP', 'AIRMASS', 'ECL_LON', 'ECL_LAT', 'SUN_LON','GAL_LAT','GAL_LON', 'AIRMASS'])
        check = np.isin(needed_params, self.ObsMeta.dtype.names)
        if np.any(check==False):
            print("Missing Required Parameters: ")
            print(needed_params[np.where(check == False)])
        else:
            pass
            #print("All parameters needed are available")



    def extinction_curve(self, X):
        ext_curve = 10**(-0.4*self.zen_ext * X)
        ext_curve[np.isnan(ext_curve)] = 0
        return ext_curve

sys.path.insert(0,'/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/python/')
import get_zodi_spectra as zodi
import get_isl_spectra as isl 
import get_scatt_moon as moon 

class SkyModel(object):
    def __init__(self, obsmeta=None):
        if obsmeta is None:
            obsmeta = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/test_ObsMeta.npy')


        self.ObsMeta = astropy.table.Table(obsmeta)

        self.Z = zodi.ZodiModel(self.ObsMeta)
        self.M = moon.MoonModel(self.ObsMeta)
        self.I = isl.ISLModel(self.ObsMeta)
        self.S = Sky(self.ObsMeta, wave_range = None)
        self.wave_range = self.S.wave_range

        self.model_components = [self.Z, self.M, self.I]

    def run_model(self):
        Model = {}
        s = []
        for mc in self.model_components:
            mc.get_spectrum()
            Model[mc.name] = mc.flux
            s.append(mc.flux)
        Model['total'] = np.sum(s, axis=0)
        Model['wave_range'] = self.wave_range
        self.Model = Model

    def plot_components(self):
        self.run_model()
        for name, spectra in self.Model.items():
            if name != 'wave_range':
                plt.plot(self.Model['wave_range'], spectra, label = name)
        plt.legend()




if __name__ == '__main__':
    SM = SkyModel()
    SM.plot_components()
    plt.show()

    #Model.test()



