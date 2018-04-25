import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.table

from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('serif')
font.set_size('large')

sys.path.insert(0,'/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/python/')
import sky_model
import get_zodi_spectra as zodi 
import get_isl_spectra as isl 
import get_scatt_moon as moon 


class RunSkyModel(object):
    def __init__(self, ObsMeta):

        self.Zodi = zodi.ZodiSpectrum(ObsMeta)
        self.ISL = isl.ISLSpectrum(ObsMeta)
        self.Moon = moon.MoonSpectrum(ObsMeta)

        self.components = [self.Zodi, self.ISL, self.Moon]
        self.wave_range = self.Zodi.wave_range

        total_flux = []
        for component in self.components:
            component.get_spectrum()
            total_flux.append(component.flux)

        self.total_flux = np.sum(total_flux, axis = 0)

    def plot_components(self):
        for component in self.components:
            plt.plot(component.wave_range, component.flux, label = component.name)

        plt.plot(self.wave_range, self.total_flux, label = 'total')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    obs_meta = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/test_ObsMeta.npy')
    Model = RunSkyModel(obs_meta)
    Model.plot_components()
    print(Model.total_flux)

