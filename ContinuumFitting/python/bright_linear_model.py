import glob, os, sys, fnmatch
from astropy.io import fits
import astropy.table
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate 
import statsmodels.api as sm

from lmfit import models, Parameters, Parameter, Model
from lmfit.models import LinearModel, ConstantModel


sys.path.insert(0,'/Users/parkerf/Research/SkyModel/BOSS_Sky/Model/python/')
from sky_model import SkyModel 

cont_files = glob.glob('/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/split_files/*_split_flux.fits')

Mhdu = fits.open('/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/good_meta_rich.fits')
MetaData = astropy.table.Table(Mhdu[1].data)
def get_specnos(meta):
    plate = meta['PLATE']
    image = meta['IMG']
    this_meta = MetaData[(MetaData['PLATE'] == plate) & (MetaData['IMG'] == image) & ((MetaData['CAMERAS'] == 'b1')|(MetaData['CAMERAS'] == 'b2'))]
    specnos = this_meta['SPECNO']
    return np.array(specnos)

wave = np.linspace(360,630,(630-360)*10)
def make_mean_cont(hdulist, specnos, wave):
    spectra = []
    for specno in specnos:
        try:
            data = hdulist[str(specno)].data
            cont = interpolate.interp1d(data['WAVE'], data['CONT'], bounds_error=False, fill_value=0)
            spectra.append(cont(wave))
        except:
            pass
    return np.mean(spectra, axis = 0)

def make_linear_model():

    Sky = SkyModel()
    Sky.run_model()

    spectra = {}
    for name in Sky.Model.keys():
        if (name == 'wave_range') | (name == 'total'):
            pass
        else:
            spectra[name] = []

    Cont_spectra = []
    for filen in cont_files:
        hdulist = fits.open(filen)
        ObsMeta = astropy.table.Table(hdulist[1].data)
        observation = ObsMeta[0]
        ThisObs = SkyModel(observation)
        ThisObs.run_model()

        specnos = get_specnos(observation)
        mc = make_mean_cont(hdulist, specnos, ThisObs.Model['wave_range'])
        try:
            a = len(mc) 
            Cont_spectra.append(mc)
            for name in spectra.keys():
                spectra[name].append(ThisObs.Model[name])
        except:
            print(filen) 

    Results = []
    for i, y in enumerate(np.array(Cont_spectra).T):
        X = []
        for name in spectra.keys():
            X.append(np.array(spectra[name]).T[i])
        X = np.vstack(X).T
        x = sm.add_constant(X)
        model = sm.OLS(y, x)
        results = model.fit()
        p = results.params
        Results.append(p)

    return Results, spectra

def test_model(filen, Results, spectra):
    Spectra = []

    hdulist = fits.open(filen)
    ObsMeta = astropy.table.Table(hdulist[1].data)
    observation = ObsMeta[0]

    ThisObs = SkyModel(observation)
    ThisObs.run_model()

    specnos = get_specnos(observation)
    mc = make_mean_cont(hdulist, specnos, ThisObs.Model['wave_range'])

    for name in spectra.keys():
        Spectra.append(ThisObs.Model[name][:-1])
        
    X = sm.add_constant(np.array(Spectra).T)
    res = np.dot(Results,X.T)
    mod = res.diagonal()
    
    return mc[:-1], mod

def plot_results(test_results):
    plt.figure()
    plt.plot(Z.wave_range[:-1], results[0], 'k-', label = 'data')
    plt.plot(Z.wave_range[:-1], results[1], label = 'fit')
    plt.legend()
    plt.xlim(360,630)


def main():
    Results, spectra = make_linear_model()
    for filen in np.random.choice(cont_files,10):
        plot_results(test_model(filen, Results, spectra))

if __name__ == '__main__':
    main()

