from lmfit import minimize, Parameters, fit_report, Minimizer
import numpy as np 


blue_vac_lines = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/files/blue_vac_lines.npy')

def clean_spectra(spectrum):
    """Takes out all nan/inf so lstsq will run smoothly
    """
    ok = ((np.isfinite(spectrum['SKY'])) & (spectrum['SKY'] > 0.))

    sky = spectrum['SKY'][ok]
    sigma = spectrum['IVAR'][ok]
    disp = spectrum['DISP'][ok]

    wave = spectrum['WAVE'][ok]
    return wave, sky
def scatter_profile(x, amplitude, center, N_eff):
        w = center/N_eff * (1/(np.sqrt(2)*np.pi))
        top = w**2.
        bot = ((x-center)**2+w**2)
        l = amplitude*top/bot
        return l

def line_profile(x, amp1, amp2, a, center, wave1, wave2, sig1, sig2, N):
    gauss1 = amp1*np.exp(-(x-wave1)**2/(2*sig1**2.))
    gauss2 = amp2*np.exp(-(x-wave2)**2/(2*sig2**2.))
    core = gauss1 + gauss2

    w = center/N_eff * (1/(np.sqrt(2)*np.pi))
    t = w**2.
    b = ((x-center)**2+w**2)
    scatter = a*t/b

    return core + scatt

def my_model(x, params):
    model = None
    for i, line in enumerate(blue_vac_lines):
        pref = 'f%s_' % str(i).zfill(4)
        amp1 = params[pref+'amp1'].value
        amp2 = params[pref+'amp2'].value
        a = params[pref+'a'].value
        center = params[pref+'center'].value
        wave1 = params[pref+'wave1'].value
        wave2 = params[pref+'wave2'].value
        sig1 = params[pref+'sig1'].value
        sig2 = params[pref+'sig2'].value
        line_model = line_profile(x, amp1, amp2, a, center, wave1, wave2, sig1, sig2, N)
        if model is None:
            model = line_model
        else:
            model = model + line_model
    Mod = model + param['c'].value
    return Mod

# def model_resids(x, params, data):
#     Mod = my_model(params, x)
#     residuals = data - Mod
#     return residuals

fit_params = Parameters()

for i, line in enumerate(blue_vac_lines):
    pref = 'f%s_' % str(i).zfill(4)
    fit_params.add(pref+'amp1', value=10, min=0)
    fit_params.add(pref+'amp2', value=10, min=0)
    fit_params.add(pref+'center', value=line, min=line-.1, max=line + .1, vary=True)
    fit_params.add(pref+'delta1', value=0.06, min=0, max=0.1, vary=True)
    fit_params.add(pref+'delta2', value=0.09, min=0, max=0.1, vary=True)
    fit_params.add(pref+'wave1', expr=pref+'center + '+pref+'delta1')
    fit_params.add(pref+'wave2', expr=pref+'center - '+pref+'delta2')

    fit_params.add(pref+'a', value=10, min=0)
    fit_params.add(pref+'sig1', value=0.1, min=0, max=1)
    fit_params.add(pref+'sig2', value=0.1, min=0, max=1)
    fit_params.add(pref+'N', value=83200, min=0)

fit_params.add('c', value = 1)

import glob
spframe_files = glob.glob('/Users/parkerf/Desktop/sample_spframe_files/*_calibrated_sky.npy')
data = np.load(spframe_files[0])
spectrum = data[25]
wave, flux = clean_spectra(spectrum)

from scipy.optimize import curve_fit

popt, pcov = curve_fit(my_model, wave, flux,p0=fit_params)



# import lmfit
# mini = lmfit.Minimizer(model_resids, fit_params, args=(wave, flux))
# result = mini.emcee(params = fit_params, workers=3)
# print(fit_report(result))

import matplotlib.pyplot as plt 
plt.plot(wave, flux, label = 'data')
plt.plot(wave, my_model(wave,*popt),label = 'fit')
#plt.plot(wave, my_model(result.params, wave), label = 'fit')
plt.legend()
plt.show()

 
#        out = model.fit(sky, params, x=wave, weights=ivar, method='leastsq', verbose=True,fit_kws={'maxfev': 2000})