#!/usr/bin/env python

"""
This program calculates the airglow lines for a model of the night sky

Title: Airglow Model
Author: P. Fagrelius
Date: 4/11/17

"""
import numpy as np
import glob
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd

class AirglowSpectrum(object):
    def __init__(self):

        self.AIRGLOW_DIR = '/Users/parkerf/Research/SkyModel/SkyModelling/AirglowSpectra/cosby/'
        self.extinction_file = '/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/files/kpnoextinct.dat'
        self.fiber_area = np.pi #arcsec**2

        print("Airglow Model Initialized")

    def get_airglow_spectra(self):
        """This function loads the airglow files from Cosby et al paper and changes the format"""

        
        AF = glob.glob(self.AIRGLOW_DIR+'/*.txt')
        AL = []
        for file in AF:
            data = pd.read_csv(file, delim_whitespace=True)
            d = data.to_records(index=False)
            AL.append(np.array(d))
        self.AirglowLines = np.hstack(AL)

    def get_significant_lines(self):
        """Gets only the airglow lines needed to make an appropriate fit. This could be
        different for the blue and red CCDs. """

        sig = np.where(self.AirglowLines['obs_eint'] > 0.1)
        self.AirglowLines = self.AirglowLines[sig]

    def air_to_vac(self, wave):
        """Index of refraction to go from wavelength in air to wavelength in vacuum
        Equation from (Edlen 1966)
        vac_wave = n*air_wave
        """
        #Convert to um
        wave_um = wave*.001
        ohm2 = (1./wave_um)**(2)

        #Calculate index at every wavelength
        nn = []
        for x in ohm2:
            n = 1+10**(-8)*(8342.13 + (2406030/float(130.-x)) + (15997/float(389-x)))
            nn.append(n)
        
        #Get new wavelength by multiplying by index of refraction
        vac_wave = nn*wave
        return vac_wave

    def get_extinction_curve(self, airmass):
        ext_df = pd.read_csv(self.extinction_file, delim_whitespace=True)
        extinction = np.exp(-ext_df['EXT'] * airmass)
        self.ext = interpolate.interp1d(ext_df['WAVE']/10.,extinction,bounds_error=False, fill_value=0)


    def airglow_spectrum(self, wave_range,alt, airmass):
        """This interpolates over the airglow lines and applies the extinction to the
        the airglow lines
        """
        self.get_airglow_spectra()
        self.get_significant_lines()
        self.get_extinction_curve(airmass)

        wave = self.air_to_vac(self.AirglowLines['obs_wave'])
        air_flux = self.AirglowLines['obs_eint']*self.fiber_area*.1*self.ext(wave)
        line = interpolate.interp1d(wave,air_flux,bounds_error=False, fill_value=0)

        Airglow = line(wave_range)#*absorption(wave_range)

        return Airglow

    def f_R_ext(Xag):
        """Rayleigh scattering for airglow from Noll et al
        """
        f_R = 1.669*np.log10(Xag)-0.146
        return f_R

    def f_M_ext(Xag):
        """Mie scattering for airglow from Noll et al
        """
        f_M = 1.732*np.log10(Xag)-0.318
        return f_M

    def extinction(alt, airmass):
        """
        Calculates the transmission due to Rayleigh and Mie scattering
        """
        z = 90-alt
        Xag = (1-0.972*np.sin(np.deg2rad(z))**2.)**(-0.5)

        fR = f_R_ext(Xag)
        fM = f_M_ext(Xag)

        tau = (fR*0.27+fM*0.01)*airmass
        trans = np.exp(-tau)
        return trans

    def test(self):
        wave_range = np.linspace(360,1040,(1040-360)*100)
        airmass = 1.003
        alt = 85.1

        AG = self.airglow_spectrum(wave_range, alt, airmass)
        plt.plot(wave_range, AG, label = 'Airglow Lines')
        plt.legend()

if __name__=="__main__":
    AG = AirglowSpectrum()
    AG.test()
    plt.show()