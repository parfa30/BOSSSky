"""
Dumping code I wrote for the Rayleigh and Mie transmissions. Would be good to get an absorption curve figured out here as well.
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd 

class Transmission(object):
	def __init__(self):

		#COnstants for transmission coefficients
       		self.p = 744 #pressure from cerro paranal (need to change)
        	self.H = 2.64 #2.788 #height
        	self.od = (self.p/1013.25)*(0.00864+self.H*6.5*10**(-6))
        	self.k0 = 0.013 #mag/AM
        	self.a = -1.38

	def rayleigh_trans(self, wave_range, airmass):   
       		trans = []
        	for wave in wave_range:
            		x = wave**(-(3.916+0.074*wave+(0.050/float(wave))))
            		odR = self.od*x
            		trans.append(np.exp(-odR*airmass))

        	return np.array(trans)

	def mie_trans(self,wave_range, airmass):

        	trans = []
        	for wave in wave_range:
            		if wave <= 0.4:
                	k = 0.05
                	trans.append((10**(-0.4*k*airmass)))
            	else: 
                	k = self.k0*wave**(self.a)
                	trans.append((10**(-0.4*k*airmass)))
        	return np.array(trans)
