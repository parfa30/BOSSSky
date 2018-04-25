import glob, os, sys, fnmatch
from astropy.io import fits
import astropy.table
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate 

cont_files = glob.glob('/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/split_files/*.fits')

Mhdu = fits.open('/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/good_meta_rich.fits')
MetaData = astropy.table.Table(Mhdu[1].data)
def get_specnos(meta):
    plate = meta['PLATE']
    image = meta['IMG']
    this_meta = MetaData[(MetaData['PLATE'] == plate) & (MetaData['IMG'] == image) & ((MetaData['CAMERAS'] == 'b1')|(MetaData['CAMERAS'] == 'b2'))]
    specnos = this_meta['SPECNO']
    return np.array(specnos)

wave = np.linspace(360,630,(630-360)*10)
def make_mean_cont(hdulist, specnos):
    spectra = []
    for specno in specnos:
        try:
            data = hdulist[str(specno)].data
            cont = interpolate.interp1d(data['WAVE'], data['CONT'], bounds_error=False, fill_value=0)
            spectra.append(cont(wave))
        except:
            pass
    return np.mean(spectra, axis = 0)

def create_mean_file(hdu_list):
    ObsMeta = astropy.table.Table(hdu_list[1].data)

    new_hdu_list = fits.HDUList()            
    new_hdu_list.append(fits.BinTableHDU(ObsMeta.as_array(), name='Meta'))

    for observation in ObsMeta:
        specnos = get_specnos(observation)
        mean_cont = make_mean_cont(hdu_list, specnos)
        model_fit = np.zeros(len(mean_cont), dtype=[('CONT', 'f8')])
        model_fit['CONT'] = np.array(mean_cont)
        new_hdu_list.append(fits.BinTableHDU(model_fit, name=str(observation['IMG'])))

    new_hdu_list.writeto('split_files/mean_cont/%d_mean_cont.fits' % int(observation['PLATE']), overwrite=True)

def main():
    for file in cont_files:
        hdulist = fits.open(file)
        create_mean_file(hdulist)


if __name__ == '__main__':
    main()
