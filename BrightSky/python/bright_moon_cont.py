
"""
Title: Bright Moon Continuum Study
Author: Parker Fagrelius
Date: Dec. 12, 2017

This code is meant to investigate the continuum levels for the DESI BGS survey.
They will be looking at the 4000A break for redshifts ~0.15. We want to know
what values to estimate for the sky background.

INPUT:  (1) Meta file for just bright sky plates/images. This was created by 
           cut_meta_data.py and makes the following cuts: 
           MOON_ALT > 0 & MOON_SEP < 90
        (2) PLATE_calibrated_sky.npy. These are created by spframe_flux.py.

OUTPUT: (1) moon_data.npy is a file with the following:
            - 'SKY_VALUE': mean sky value between 460-480nm
            - MOON_ALT, MOON_SEP, AIRMASS, DAYS2FULL for that particular image
        (2) data evaluation plots, making histogram plots for the parameters
            mentioned above
        (3) linear plots to show visually the dependence of the sky values on
            each parameter.
        (4) Linear regression using all four parameters. Output lists the following
            coefficients:
            x1 = MOON_ALT; x2 = MOON_SEP; x3 = AIRMASS; x4 = DAYS2FULL

The functions should be run in self.main() like this:
self.get_data()
self.get_cont()
self.linear_regression()
self.data_breakdown()
self.plot_linear_relations()
"""

import numpy as np 
import matplotlib.pyplot as plt 
import glob, os, sys
import statsmodels.api as sm 
import pandas as pd
from astroplan import Observer
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import EarthLocation

from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('serif')
font.set_size('small')

class BrightSkyCont(object):
    def __init__(self):
        #Identify which meta file to use
        self.bright_meta = "/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/moon_meta_rich.npy"
        self.dark_meta = "/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/dark_meta_rich.npy"
        self.good_meta = "/Users/parkerf/Research/SkyModel/BOSS_Sky/Analysis/MetaData/good_meta_rich.npy"

        #Identify directory of sky flux files
        self.spframe_dir = "/Volumes/PFagrelius_Backup/sky_data/sky_flux/"

        #This sets the window for finding the mean continuum value
        self.lower = 460
        self.upper = 480

        self.feat = ['SKY_VALUE','MOON_ILL','MOON_PHASE','FIBER_RA', 'FIBER_DEC', 'MJD', 'TAI-BEG', 'TAI-END', 
                    'RA', 'DEC', 'AIRMASS', 'ALT', 'AZ', 'EXPTIME', 'SEEING20', 'SEEING50', 'SEEING80',
                     'AIRTEMP', 'DEWPOINT', 'DUSTA', 'DUSTB', 'WINDD25M', 'WINDS25M', 'GUSTD', 'GUSTS', 'HUMIDITY', 
                     'PRESSURE', 'WINDD', 'WINDS', 'MOON_LAT', 'MOON_LON', 'SUN_LAT', 'SUN_LON', 'MOON_ALT', 'MOON_AZ', 
                     'SUN_ALT', 'SUN_AZ', 'MOON_D', 'MOON_SEP', 'SUN_MOON_SEP', 'SUN_ELONG', 'DAYS2FULL', 'ECL_LAT', 'ECL_LON', 
                     'GAL_LAT', 'GAL_LON', 'AZ_CALC', 'FLI', 'SEASON', 'HOUR', 'SOLARFLUX', 'PLATE','IMG']

        self.lin_regress_feat = ['SKY_VALUE','MOON_ILL','MOON_ALT','MOON_SEP','AIRMASS','HOUR', 'SUN_AZ',
                    'SUN_SEP', 'SEEING50']

        self.dark_mean = 2.78      

        self.apache = Observer.at_site('Apache Point Observatory')

    def make_data_file(self, name = 'bright', get_outliers = True):
        spframe_files = glob.glob(self.spframe_dir+"/*_calibrated_sky.npy")
        spframe_plates = np.array([int(filen[-23:-19]) for filen in spframe_files])

        self.meta = np.load(self.good_meta)
        print("meta data is loaded")

        self.plate_list = np.load("%s_list.npy"%name)
        plates_ = np.unique(self.plate_list['PLATE'])
        plates = [x for x in plates_ if x != '7339']
        self.avail_plates = np.intersect1d(plates, spframe_plates)
        print("Available plates: ", self.avail_plates)

        print("Ready to start measuring the continuum between %d and %d"%(self.lower, self.upper))
        bad_plates = self.get_cont(cont_file_name='all_%s'%name)
        np.save('bad_%s_plates'%name, bad_plates)
        full_data = np.load('all_%s_data.npy'%name)
        if get_outliers:
            df = pd.DataFrame(full_data)
            good_df = df[~df.PLATE.isin(bad_plates)]
            np.save('%s_data'%name, good_df.to_records(index=False))

    def plot_mean_dark(self):
        data = np.load('dark_data.npy')

        plt.figure()
        plt.plot(data['MOON_ALT'],data['SKY_VALUE'],  'x')
        plt.axhline(np.mean(data['SKY_VALUE']), label = np.mean(data['SKY_VALUE']))
        plt.legend()
        plt.ylabel("Flux $10^{-17} erg/cm^{2}/s/\AA$", fontproperties = font)
        plt.xlabel("Moon Alt (deg)", fontproperties = font)
        plt.title("Sky continuum flux for Dark Time", fontproperties = font)
        plt.savefig("dark_time_mean.png")

    def make_bright_list(self):
        self.get_data(self.good_meta, get_outliers = True, name = 'good', bad_name = 'good_bad')
        data = np.load('good_data.npy')
        gray_level = 2.5*self.dark_mean
        gray = np.where((data['MOON_ALT']>0)&(data['SKY_VALUE']<gray_level))
        bright = np.where((data['MOON_ALT']>0)&(data['SKY_VALUE']>gray_level))

        #Save these lists
        np.save('gray_list', data[gray])
        np.save('bright_list', data[bright])

        #Plot
        plt.figure()
        plt.plot(data['MOON_ALT'], data['SKY_VALUE'], 'b.', label = 'dark')
        plt.plot(data['MOON_ALT'][gray], data['SKY_VALUE'][gray], 'g.', label = 'gray')
        plt.plot(data['MOON_ALT'][bright], data['SKY_VALUE'][bright], 'r.', label = 'bright')
        plt.legend(loc='upper left', prop=font)
        plt.xlabel("Moon Altitude (deg)", fontproperties = font)
        plt.ylabel("Flux $10^{-17} erg/cm^{2}/s/\AA$", fontproperties = font)
        plt.title("Sky continuum flux (460-480 nm)")
        plt.savefig('dark_gray_bright.png')

    def davids_plot(self):
        data = np.load('good_data.npy')
        red = np.where(data['MOON_ALT'] > 20)
        green = np.where((data['MOON_ALT']>20)&(data['MOON_SEP']>50))
        blue = np.where((data['MOON_ALT']>20)&(data['MOON_SEP']>80))

        plt.figure()
        plt.plot(data['SUN_ALT'], data['SKY_VALUE'], 'k.',ms=2)
        plt.xlabel("Sun angle above the horizon (deg)", fontproperties=font)
        plt.ylabel("Flux $10^{-17} erg/cm^{2}/s/\AA$", fontproperties = font)
        plt.savefig('david_plot_1.png')

        plt.figure()
        plt.plot(data['MOON_ILL'], data['SKY_VALUE'], 'k.')
        plt.plot(data['MOON_ILL'][red], data['SKY_VALUE'][red], 'r.', label = 'moon alt > 20')
        plt.plot(data['MOON_ILL'][green], data['SKY_VALUE'][green], 'g.', label = 'moon alt > 20, moon sep > 50')
        plt.plot(data['MOON_ILL'][blue], data['SKY_VALUE'][blue], 'b.', label = 'moon alt > 20, moon sep > 80')
        plt.xlabel("Moon Phase (1 = full)", fontproperties=font)
        plt.ylabel("Flux $10^{-17} erg/cm^{2}/s/\AA$", fontproperties = font)
        plt.legend(loc='upper left', prop=font)
        plt.savefig('david_plot_2.png')


    def data_histograms(self):
        """This function plots histograms of the parameters
        """
        df = pd.DataFrame(self.data)
        print("Number of data points: ", len(self.data))
        
        for feat in self.lin_regress_feat[1:]:
            plt.figure()
            df[feat].hist(bins=20, alpha = 0.5) #,facecolor = 'blue', alpha = 0.5)
            plt.title(feat, fontproperties = font)
            plt.savefig('plots/%s_brightsky_hist.png'%feat)

    def get_cont(self, cont_file_name = 'bright'):
        """This function measures the mean continuum value in the windown given in the initialization. It also saves
        this value and the meta data in a file moon_data.npy
        """
        mdtype = [tuple((x,'f8')) for x in self.feat]
        moon_data = []
        all_std = []
        for plate in self.avail_plates:
            plate = int(plate)
            print(plate)
            try:
                data = np.load(self.spframe_dir+"%d_calibrated_sky.npy"%plate)
                plate_means = []

                plate_meta = self.meta[self.meta['PLATE'] == plate]
                images = np.unique(plate_meta['IMG'])

                image_mean = []
                for image in images:
                    image = int(image)
                    #idx = np.where((plate_meta['IMG'] == image)&())
                    image_meta = plate_meta[plate_meta['IMG']==image]
                    blue_meta = image_meta[(image_meta['CAMERAS'] == b'b1') | (image_meta['CAMERAS'] == b'b2')]
                    for specno in np.unique(blue_meta['SPECNO']):
                        
                        idx = np.where((data[specno]['WAVE']>self.lower)&(data[specno]['WAVE']<self.upper))
                        mean_sky_value = np.mean(data[specno]['SKY'][idx])
                        image_mean.append(mean_sky_value)
                        plate_means.append(mean_sky_value)

                    mydata = []
                    sky_value = np.mean(image_mean)
                    this_meta = image_meta[0]

                    #To avoid zero values
                    if sky_value > 0.1:
                        date = this_meta['MJD']
                        t = Time(date, scale='tai', format = 'mjd')
                        moon_phase = self.apache.moon_phase(t).value
                        moon_ill = self.apache.moon_illumination(t)

                        mydata = [sky_value, moon_ill, moon_phase]

                        for x in self.feat[3:]:
                            mydata.append(this_meta[x])
                        moon_data.append(tuple(mydata))

                all_std.append([plate, np.std(plate_means)])
            except:
                print("something went wrong")

        np.save('%s_data'%cont_file_name,np.array(moon_data,dtype=mdtype))
        outliers = np.where(np.array(all_std)[:,1] > 1)
        return np.array(all_std)[outliers][:,0]
    
    def plot_linear_relations(self, name):

        for feat in self.lin_regress_feat[1:]:
            plt.figure()
            plt.plot(self.data[feat], self.data['SKY_VALUE'], '.')
            plt.title("Mean Continuum Level at 460nm vs. %s" % feat, fontproperties = font)
            plt.ylabel("Flux $10^{-17} erg/cm^{2}/s/\AA$", fontproperties = font)
            if (feat == 'MOON_ALT') | (feat == 'MOON_SEP'):
                plt.xlabel("Degrees",fontproperties = font)
            elif feat == 'AIRMASS':
                plt.xlabel("Airmass",fontproperties = font)
            elif feat == 'DAYS2FULL':
                plt.xlabel("Days until full moon",fontproperties = font)
            plt.savefig("plots/%s_linear_plots_%s.png"%(feat, name))

    def linear_regression(self):

        X = []
        for feat in self.lin_regress_feat[1:]:
            X.append(self.data[feat])
        X = np.column_stack(X)

        self.y = self.data['SKY_VALUE']
        print(X.shape, self.y.shape)

        sm.OLS.exog_names = self.feat[1:-2]
        results = sm.OLS(self.y, X).fit()

        self.params = results.params
        self.model = np.dot(X, self.params)

        print(results.summary())

    def res_plot(self):
        plt.figure()
        plt.hist2d(self.model, self.y-self.model,bins=(50,50), cmap=plt.cm.jet)
        plt.colorbar()
        plt.xlabel("Model flux",fontproperties=font)
        plt.ylabel("Data - Model (residuals)",fontproperties=font)
        plt.savefig("plots/residual_plot.png")

    def make_fits_table(self,filen):
        name = "%s.fits"%os.path.splitext(filen)[0]
        print(name)
        if os.path.isfile(name):
            print("file already exists. Going to delete and create again.")
            os.remove(name)
        
        data = np.load(filen)
        t = Table(data)
        t.write(name, format='fits')


    def run_regression(self):

        #self.plot_linear_relations(name='with_scatter')
        #plt.show()
        self.data = np.load('bright_data.npy')
        self.linear_regression()
        self.data_histograms()
        self.plot_linear_relations(name='no_scatter')
        self.res_plot()
        plt.show()

    def main(self):
        self.make_data_file(name = 'dark', get_outliers = True)
        self.make_fits_table('dark_data.npy')
        #self.get_data(self.bright_meta, get_outliers = True, name = 'test', bad_name = 'test')
        #self.run_regression()
        #self.make_bright_list()
        # self.get_dark_sky()
        #self.davids_plot()
        #self.run_regression()
        #plt.show()
    

if __name__ == '__main__':
    BS = BrightSkyCont()
    BS.main()

