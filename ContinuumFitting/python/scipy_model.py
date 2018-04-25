import numpy as np 

blue_vac_lines = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/FitSpectra/files/blue_vac_lines.npy')

def scatter_profile(x, amplitude, center, N_eff):
    w = center/N_eff * (1/(np.sqrt(2)*np.pi))
    top = w**2.
    bot = ((x-center)**2+w**2)
    l = amplitude*top/bot
    return l

def my_profile(x, *args):
    amp1, amp2, a, center, wave1, wave2, sig1, sig2, N = list(args[0])
    gauss1 = amp1*np.exp(-(x-wave1)**2/(2*sig1**2.))
    gauss2 = amp2*np.exp(-(x-wave2)**2/(2*sig2**2.))
    core = gauss1 + gauss2

    scatt = scatter_profile(x, a, center, N)
    return core + scatt

def my_model(wave, *P):
    model = None
    for i, line in enumerate(blue_vac_lines):
        line_model = my_profile(wave, np.array(P)[i*9:(i+1)*9])
        if model is None:
            model = line_model
        else:
            model = model + line_model
    return model + P[-1]


P = []       
for line in blue_vac_lines:
    P.append([10,10,line,.06,.09,10,0.1,0.1,83200])
P.append(1)
P = np.hstack(P)

from scipy.optimize import curve_fit
import glob
spframe_files = glob.glob('/Users/parkerf/Desktop/sample_spframe_files/*_calibrated_sky.npy')
data = np.load(spframe_files[0])
spectrum = data[25]
xdata = spectrum['WAVE']
ydata = spectrum['SKY']

popt, pcov = curve_fit(my_model, xdata, ydata,p0=P,check_finite=False)
import matplotlib.pyplot as plt 

plt.plot(xdata,ydata,label = 'data')
plt.plot(xdata, my_model(xdata,*popt),label = 'fit')
plt.legend()
plt.show()


