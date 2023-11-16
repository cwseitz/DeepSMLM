import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
from pathlib import Path
from SMLM.localize import LoGDetector

class PipelineLifetime:
    def __init__(self,config):
        self.config = config
        self.datapath = config['datapath']
        
    def forward(self,stack,ax=None,thresh=0.001,det_idx=0,bin_thresh=0.2,window_size=8):
        mx = np.max(stack,axis=0)
        det_frame = stack[det_idx]
        log = LoGDetector(det_frame,threshold=thresh,blob_markersize=1)
        spots = log.detect(); log.show(100*det_frame,ax=ax); plt.show()
        xidx = spots['x'].to_numpy().astype(np.int16)
        yidx = spots['y'].to_numpy().astype(np.int16)
        vals = stack[:,xidx,yidx]
        nt,nspots = vals.shape
        vals = self.normalize(vals,window_size)
        binary = self.binarize(vals,bin_thresh)
           
        return vals, binary

    def binarize(self,data,thresh):
        nt,nspots = data.shape
        thresh = thresh*np.ones((nspots,))
        thresh_vals = data > thresh.reshape(1,-1) 
        binary = thresh_vals.astype(int)
        return binary
            

    def normalize(self,data,window_size):
        kernel = np.ones(window_size) / window_size
        out = []
        for n in range(data.shape[1]):
            y = data[:,n]/data[:,n].max()
            out.append(y)
        return np.array(out).T

    def plot_intensity(self,vals,binary):    
        nt,nspots = vals.shape
        fig,ax = plt.subplots(nspots,1,figsize=(3*nspots,3))
        for n in range(nspots):
            ax[n].plot(vals[:,n])
            ax[n].plot(binary[:,n])
        plt.tight_layout()
            
    def plot_lifetimes(self,binary,dt):
        nt,nspots = binary.shape
        all_off_times = []; all_on_times = []
        for n in range(nspots):
            off_times, on_times = self.lifetime(binary[:,n],dt)
            all_off_times += off_times; all_on_times += on_times
            
        mx = max(on_times+off_times); mx = 250
        bins = np.arange(0,mx,dt)        
        fig,ax = plt.subplots(1,2,figsize=(5,3))
        ax[0].hist(off_times,bins=bins,color='black')
        ax[0].set_xlabel('OFF lifetime')
        ax[1].hist(on_times,bins=bins,color='black')
        ax[1].set_xlabel('ON lifetime')
        plt.tight_layout()

    def lifetime(self,X,dt):
        X1 = X
        X2 = np.logical_not(X)
        off_times = []; on_times = []
        x1 = np.argwhere(X1 == 1).flatten()
        x2 = np.argwhere(X2 == 1).flatten()
        diff1 = np.diff(x1)
        diff2 = np.diff(x2)
        diff1 = diff1[np.argwhere(diff1 >= 2)] - 1
        diff2 = diff2[np.argwhere(diff2 >= 2)] - 1
        off_times += list(np.squeeze(diff2)*dt)
        on_times += list(np.squeeze(diff1)*dt)
        return off_times, on_times
        
        
