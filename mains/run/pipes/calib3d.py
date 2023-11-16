import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

class PipelineCalibrate3D:
    def __init__(self,config,dataset):
        self.config = config
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack

    def fitgaussian1d(self, x_data, y_data, initial_guess=None):

        def gaussian(x, amplitude, mean, stddev, b):
            return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2)) + b

        if initial_guess is None:
            initial_guess = [max(y_data), np.mean(x_data), 1.0, 0.0]

        params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
        return params, covariance

    def calibrate(self,row,col,plot_iters=False):
        gxz = self.stack[:,row,:]
        gyz = self.stack[:,:,col]
        nz,ns = gxz.shape
        space = np.arange(0,ns,1)
        sigmax = []; sigmay = []
        for n in range(nz):
            px, covx = self.fitgaussian1d(space,gxz[n,:])
            xprofile = px[0] * np.exp(-(space - px[1])**2 / (2 * px[2]**2)) + px[3]
            py, covy = self.fitgaussian1d(space,gyz[n,:])
            yprofile = py[0] * np.exp(-(space - py[1])**2 / (2 * py[2]**2)) + py[3]
            sigmax.append(px[2]); sigmay.append(py[2])
            if plot_iters:
                fig,ax=plt.subplots(1,2)
                ax[0].plot(space,gxz[n,:])
                ax[0].plot(space,xprofile)
                ax[1].plot(space,gyz[n,:])
                ax[1].plot(space,yprofile)
                plt.show()
          
        fig,ax=plt.subplots(figsize=(3,3)) 
        space = np.linspace(-1,1,nz)
        fity = np.polyfit(space,sigmay,3)
        fitx = np.polyfit(space,sigmax,3)
        print(fitx,fity)
        polyy = np.poly1d(fity)
        polyx = np.poly1d(fitx)
        ax.scatter(space,sigmax,color='cornflowerblue',label='x')
        ax.scatter(space,sigmay,color='blue',label='y')
        ax.plot(space,polyy(space),color='blue')
        ax.plot(space,polyx(space),color='cornflowerblue')
        ax.set_xlabel('z (um)')
        ax.set_ylabel(r'$\sigma$')
        ax.legend()
        plt.tight_layout()
        plt.show()

