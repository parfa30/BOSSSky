#!/usr/bin/env python
"""
This program essentially takes the place of 'fit_spectra.py'.

This fits SpFrame spectra with airglow lines and a simple model for the continuum.
The fit is done with a nonlinear fit of a handful of airglow lines with a profile of
two gaussians and a lorentzian for the scattered light from the VPH grating.

INPUT: * SpFrame flux files as .npy files with wavelength and sky spectra

OUTPUT: fits files arranged by plate number: '####_split_flux.fits'.
        The fits file will be organized as follows:
        * HDU[1] includes the meta data for each observation in the fits file
        * The remaining HDUs are named by their associated SPECNO that is in the MetaData
          files and identifies each spectrum with their spframe flux counter part
        * Each remaining HDU includes the following: WAVE, CONT, LINES, FLUX


Title: SpFrame Flux Continuum Fit
Author: P. Fagrelius
Date: March, 21018

export OMP_NUM_THREADS=1 (32/#) multiprocessing)

MPI - multiprocessing across nodes. not great for python data parallelization. steep learnign curve
Quequ do. 

"""

import os, sys
import glob
import numpy as np
import multiprocessing
from datetime import datetime

import astropy.table
from astropy.io import fits

from lmfit import Model
from lmfit.models import ConstantModel

import matplotlib.pyplot as plt

        
SAVE_DIR = '/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/split_files/dark_blue/'#+datetime.strftime(datetime.now(),'%Y_%m_%d_%H:%M')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


SPDATA_DIR = '/Volumes/PFagrelius_Backup/sky_data/sky_flux/' #Spframe flux files
blue_vac_lines = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/files/blue_vac_lines.npy') #List of airglow lines converted to vacuum wavelength

#Meta data
Mhdu = fits.open('/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/good_meta_rich.fits')
Mhdu_obs = fits.open('/Users/parkerf/Research/SkyModel/BOSS_Sky/BrightSky/data/good_data.fits')
SpecData = astropy.table.Table(Mhdu[1].data)
MetaData = astropy.table.Table(Mhdu_obs[1].data)
Mhdu.close()
Mhdu_obs.close()


def fit_sky_spectrum(inputs):

    specno, spectrum = inputs
    print('SPECNO: ', specno)

    ## Clean Spectrum
    ok = ((np.isfinite(spectrum['SKY'])) & (spectrum['SKY'] > 0.))

    sky = spectrum['SKY'][ok]
    ivar = spectrum['IVAR'][ok]
    disp = spectrum['DISP'][ok]
    wave = spectrum['WAVE'][ok]
    
    ## Line Model
    def scatter_profile(x, amplitude, center, N_eff):
        w = center/N_eff * (1/(np.sqrt(2)*np.pi))
        top = w**2.
        bot = ((x-center)**2+w**2)
        l = amplitude*top/bot
        return l

    def my_profile(x, amp1, amp2, a, center, wave1, wave2, sig1, sig2, N):
        gauss1 = amp1*np.exp(-(x-wave1)**2/(2*sig1**2.))
        gauss2 = amp2*np.exp(-(x-wave2)**2/(2*sig2**2.))
        core = gauss1 + gauss2

        scatt = scatter_profile(x, a, center, N)
        return core + scatt

    ## Run Model
    mod = None
    for i, line in enumerate(blue_vac_lines):
        pref = 'f%s_' % str(i).zfill(4)
        line_model = Model(my_profile, prefix=pref)
        line_model.set_param_hint(pref+'amp1', value=10, min=0)
        line_model.set_param_hint(pref+'amp2', value=10, min=0)
        line_model.set_param_hint(pref+'center', value=line, min=line-.1, max=line + .1, vary=True)
        line_model.set_param_hint(pref+'delta1', value=0.06, min=0, max=0.1, vary=True)
        line_model.set_param_hint(pref+'delta2', value=0.09, min=0, max=0.1, vary=True)
        line_model.set_param_hint(pref+'wave1', expr=pref+'center + '+pref+'delta1')
        line_model.set_param_hint(pref+'wave2', expr=pref+'center - '+pref+'delta2')
    
        line_model.set_param_hint(pref+'a', value=10, min=0)
        line_model.set_param_hint(pref+'sig1', value=0.1, min=0, max=1)
        line_model.set_param_hint(pref+'sig2', value=0.1, min=0, max=1)
        line_model.set_param_hint(pref+'N', value=83200, min=0)

        if mod is None:
            mod = line_model
        else:
            mod = mod + line_model

    offset = ConstantModel()
    offset.set_param_hint('c', value=1)

    model = mod + offset
    params = model.make_params()
    
    try: #Multiprocessing won't do this fit
        
        out = model.fit(sky, params, x=wave, weights=ivar, method='leastsq', verbose=True,fit_kws={'maxfev': 2000})
        comps = out.eval_components(x=wave) 

        cont = sky.copy()
        Lines = []
        for name, comp in comps.items():
            if name == 'constant':
                pass
            else:
                idx = np.where(comp > 10**-1)[0]
                cont[idx] = 0
                Lines.append(comp)
        lines = np.sum(Lines, axis=0)

        idx = np.where(cont <= 0)
        groups = np.split(idx[0], np.where(np.diff(idx[0]) != 1)[0]+1)

        for group in groups:
            first = group[0]
            last = group[-1]
            try:
                mean_sky = np.mean([cont[first-5:first-2], cont[last+2:last+5]])
                cont[first-2:last+2] = mean_sky
            except:
                mean_sky = np.mean(cont[first-5:first-2])
                cont[group] = mean_sky

        model_fit = np.zeros(len(cont), dtype=[('WAVE', 'f8'), ('LINES', 'f8'), ('CONT', 'f8'), ('FLUX', 'f8')])
        model_fit['WAVE'] = np.array(wave)
        model_fit['LINES'] = np.array(lines)
        model_fit['CONT'] = np.array(cont)
        model_fit['FLUX'] = np.array(sky)
        print("finished fit")
        return fits.BinTableHDU(model_fit, name=str(specno))

    except:
        print("ERROR!")
        #out = 'failed'
        #pass

def fit_plate_continuum(plate):
    """
    1) Identifies the observation meta data
    2) Identifies the SPECNOS for each observation
    3) Runs fit_and_split for each spectrum
    4) Saves file with split flux for the plate
    """   

    print("Running Fit and Split for Plate %d" % int(plate))

    data = np.load(SPDATA_DIR+'/%s_calibrated_sky.npy' % str(int(plate)))

    hdu_list = fits.HDUList()            
    ObsMeta = MetaData[MetaData['PLATE']==int(plate)]
    SpecnoMeta = SpecData[SpecData['PLATE']==int(plate)]
    hdu_list.append(fits.BinTableHDU(ObsMeta.as_array(), name='Meta'))

    plate_spectra = [[line['SPECNO'],data[line['SPECNO']]] for line in SpecnoMeta]
    print("Splitting continuum of %d spectra" % len(plate_spectra))

    R = []
    for spectrum in plate_spectra:
        R.append(fit_sky_spectrum(spectrum))
    for r in R:
        hdu_list.append(r)
         
    hdu_list.writeto(SAVE_DIR+'/%d_split_flux.fits' % int(plate), overwrite=True)


def main():
    filen = '/Users/parkerf/Research/SkyModel/BOSS_Sky/BrightSky/data/dark_data.fits'
    data = astropy.table.Table.read(filen)

    # spframe_files = glob.glob(SPDATA_DIR+"*_calibrated_sky.npy")
    # plates = [int(filen[-23:-19]) for filen in spframe_files]
    # print(plates)
    plates = np.unique(data['PLATE'])

    #Check which files have been completed
    print("Using directiory %s" % SAVE_DIR)
    Complete_files = glob.glob(SAVE_DIR+"/*_split_flux.fits")
    Completed_Plates = [int(os.path.split(filen)[1][0:4]) for filen in Complete_files]
    print("Completed Plates: ",Completed_Plates)

    plates_needed = np.array([p for p in plates if p not in Completed_Plates])
    print("Plates still needed: (%d)"%len(plates_needed),plates_needed)

    for plate in plates_needed:
       fit_plate_continuum(plate)

    # pool = multiprocessing.Pool(processes=2)
    # returns = pool.map(fit_plate_continuum, plates_needed)
    # pool.terminate()

        
if __name__ == '__main__':
    main()
    

