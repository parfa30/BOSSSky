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

"""

import os
import glob
import numpy as np
import multiprocessing
from datetime import datetime

import astropy.table
from astropy.io import fits

from lmfit import Model
from lmfit.models import ConstantModel

class FitCont(object):
    def __init__(self, Obs_Dict, save_dir = None):
        

        #Create directory to save the data in
        if save_dir is None:
            self.save_dir = '/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/split_files/'+datetime.strftime(datetime.now(),'%Y_%m_%d_%H:%M')
        else:
            self.save_dir = '/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/split_files/'+save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.Obs_Dict = Obs_Dict #This dictionary includes the plates and images want to fit.

        self.DATA_DIR = '/Volumes/PFagrelius_Backup/sky_data/sky_flux/' #Spframe flux files
        self.blue_vac_lines = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/files/blue_vac_lines.npy') #List of airglow lines converted to vacuum wavelength

        #Meta data
        Mhdu = fits.open('/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/good_meta_rich.fits')
        Mhdu_obs = fits.open('/Users/parkerf/Research/SkyModel/BOSS_Sky/BrightSky/data/good_data.fits')
        self.Specno_Meta = astropy.table.Table(Mhdu[1].data)
        self.MetaData = astropy.table.Table(Mhdu_obs[1].data)
        Mhdu.close()
        Mhdu_obs.close()

    def clean_spectra(self, spectrum):
        """Takes out all nan/inf so lstsq will run smoothly
        """
        ok = ((np.isfinite(spectrum['SKY'])) & (spectrum['SKY'] > 0.))

        sky = spectrum['SKY'][ok]
        sigma = spectrum['IVAR'][ok]
        disp = spectrum['DISP'][ok]

        wave = spectrum['WAVE'][ok]
        return [wave, sky, sigma, disp]

    def make_line_model(self, i, line):
        """It creates a non-linear model for an airglow line consisting
        of two gaussians and a lorentzian. It uses lmfit.
        """
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

        pref = 'f%s_' % str(i).zfill(4)
        model = Model(my_profile, prefix=pref)
        model.set_param_hint(pref+'amp1', value=10, min=0)
        model.set_param_hint(pref+'amp2', value=10, min=0)
        model.set_param_hint(pref+'center', value=line, min=line-.1, max=line + .1, vary=True)
        model.set_param_hint(pref+'delta1', value=0.06, min=0, max=0.1, vary=True)
        model.set_param_hint(pref+'delta2', value=0.09, min=0, max=0.1, vary=True)
        model.set_param_hint(pref+'wave1', expr=pref+'center + '+pref+'delta1')
        model.set_param_hint(pref+'wave2', expr=pref+'center - '+pref+'delta2')
        
        model.set_param_hint(pref+'a', value=10, min=0)
        model.set_param_hint(pref+'sig1', value=0.1, min=0, max=1)
        model.set_param_hint(pref+'sig2', value=0.1, min=0, max=1)
        model.set_param_hint(pref+'N', value=83200, min=0)

        return model

    def model_fit(self, wave, flux, ivar):
        """Builds model with a profile of each airglow line and adds a constant
        term.
        """
        mod = None
        for i, line in enumerate(self.blue_vac_lines):
            this_mod = self.make_line_model(i, line)
  
            if mod is None:
                mod = this_mod
            else:
                mod = mod + this_mod

        offset = ConstantModel()
        offset.set_param_hint('c', value=1)

        model = mod + offset
        params = model.make_params()
        
        try: #Multiprocessing won't do this fit
            out = mod.fit(flux, params, x=wave, weights=ivar, method='leastsq', fit_kws={'maxfev': 2000})      
        except:
            print("fit didn't work")
            out = 'failed'

        return out

    def fit_and_split(self, wave, flux, ivar):
        """Performs the actual fit and then pulls out the continuum part 
        and the lines from the fit.
        """
        out = self.model_fit(wave, flux, ivar)
        if out == 'failed':
            Cont, Lines, Centers = [None, None, None]
        else:
            comps = out.eval_components(x=wave)
            
            Cont = flux.copy()
            Lines = []
            for name, comp in comps.items():
                if name == 'constant':
                    pass
                else:
                    idx = np.where(comp > 10**-1)[0]
                    Cont[idx] = 0
                    Lines.append(comp)

            Lines = np.sum(Lines, axis=0)

            idx = np.where(Cont <= 0)
            groups = np.split(idx[0], np.where(np.diff(idx[0]) != 1)[0]+1)

            for group in groups:
                first = group[0]
                last = group[-1]
                try:
                    mean_sky = np.mean([Cont[first-5:first-2], Cont[last+2:last+5]])
                    Cont[first-2:last+2] = mean_sky
                except:
                    mean_sky = np.mean(Cont[first-5:first-2])
                    Cont[group] = mean_sky

            Centers = []
            for comp in comps.keys():
                if comp is not 'constant':
                    Centers.append(np.round(out.params[comp+"center"].value, 3))

        return Cont, Lines, np.array(Centers)

    def fit_plate(self, plate):
        """
        1) Identifies the observation meta data
        2) Identifies the SPECNOS for each observation
        3) Runs fit_and_split for each spectrum
        4) Saves file with split flux for the plate
        """
        print("Running Fit and Split for Plate %d with %d images" % (int(plate), len(self.Obs_Dict[plate])))
        start_plate = datetime.now()
        data = np.load(self.DATA_DIR+'/%s_calibrated_sky.npy' % str(int(plate)))

        hdu_list = fits.HDUList()            
        ObsMeta = np.vstack([self.MetaData[self.MetaData['IMG']==int(image)] for image in self.Obs_Dict[plate]])
        hdu_list.append(fits.BinTableHDU(ObsMeta, name='Meta'))

        for i_num, img in enumerate(np.unique(ObsMeta['IMG'])):
            print("Image %d", %i_num)
            ImgMeta = self.Specno_Meta[(self.Specno_Meta['IMG'] == int(img)) & ((self.Specno_Meta['CAMERAS'] == 'b1') | (self.Specno_Meta['CAMERAS'] == 'b2'))]
            #print(len(ImgMeta))
            for i, line in enumerate(ImgMeta):
                specno = line['SPECNO']
                print("%d: %d/%d" %(int(plate), i, len(ImgMeta)))
                start_line = datetime.now()
                spectrum = data[specno]
                wave, flux, ivar, sigma = self.clean_spectra(spectrum)
                cont, lines, centers = self.fit_and_split(wave, flux, ivar)
                if cont is None:
                    pass
                else:
                    model_fit = np.zeros(len(cont), dtype=[('WAVE', 'f8'), ('LINES', 'f8'), ('CONT', 'f8'), ('FLUX', 'f8')])
                    model_fit['WAVE'] = np.array(wave)
                    model_fit['LINES'] = np.array(lines)
                    model_fit['CONT'] = np.array(cont)
                    model_fit['FLUX'] = np.array(flux)
                    hdu_list.append(fits.BinTableHDU(model_fit, name=str(specno)))
                print("Line took %.2f to run"% (datetime.now()-start_line).seconds)

        hdu_list.writeto(self.save_dir+'/%d_split_flux.fits' % int(plate), overwrite=True)
        print("Plate takes %.2f sec to run" % (datetime.now()-start_plate).seconds)


    def run(self):

        plates = self.Obs_Dict.keys()

        #WOULD LIKE TO DO EVERYTHING BELOW IN PARALLEL

        #Check which files have been completed
        print("Using directiory %s" % self.save_dir)
        Complete_files = glob.glob(self.save_dir+"/*_split_flux.fits")
        Completed_Plates = [int(os.path.split(filen)[1][0:4]) for filen in Complete_files]
        print("Completed Plates: ",Completed_Plates)

        plates_needed = np.array([p for p in plates if p not in Completed_Plates])
        print("Plates still needed: (%d)"%len(plates_needed),plates_needed)

        num = len(plates_needed)//4
        PLATES = [plates_needed[:num], plates_needed[num:num*2], plates_needed[num*2:num*3], plates_needed[num*3:]]

        for plate in plates_needed:
            self.fit_plate(plate)

        # PLATES = np.unique(BrightMeta['PLATE'])
        # pool = multiprocessing.Pool(processes=2)
        # ret = pool.map(save_plate_data, PLATES)
        # pool.terminate()

def get_bright_data(self):
    """This function is used to identify only bright data if that's what you want to run.
    It also adds some additional meta data to the files.
    """
    filen = '/Users/parkerf/Research/SkyModel/BOSS_Sky/BrightSky/data/bright_data.fits'
    data = astropy.table.Table.read(filen)

    moon_zenith = 90-data['MOON_ALT']
    data['MOON_ZENITH'] = astropy.table.Column(moon_zenith.astype(np.float32), unit='deg')
    # Compute the pointing zenith angle in degrees.
    obs_zenith = 90 - data['ALT']
    data['OBS_ZENITH'] = astropy.table.Column(obs_zenith.astype(np.float32), unit='deg')

    mphase = np.arccos(2 * data['MOON_ILL'] - 1) / np.pi
    data['MPHASE'] = astropy.table.Column(mphase.astype(np.float32))

    #gray_level = 2.5*2.79
    #bright = np.where((data['MOON_ALT']>0)&(data['SKY_VALUE']>gray_level))

    bright_data = data
    bright_data['x'] = astropy.table.Column(np.linspace(0, len(bright_data)-1, len(bright_data)))
    bright_data.add_index('x')

    self.BrightMeta = bright_data

def get_dark_data():
    """This function is used to identify only bright data if that's what you want to run.
    It also adds some additional meta data to the files.
    """
    filen = '/Users/parkerf/Research/SkyModel/BOSS_Sky/BrightSky/data/dark_data.fits'
    data = astropy.table.Table.read(filen)

    DD = {}
    for plate in np.unique(data['PLATE']):
        DD[plate] = data[data['PLATE'] == plate]['IMG']
    print("Got Dark Data!")
    return DD
        
if __name__ == '__main__':
    this_dict = get_dark_data()
    FC = FitCont(this_dict, save_dir = 'dark_blue')
    FC.run()
    

