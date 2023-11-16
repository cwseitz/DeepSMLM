import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import trackpy as tp
import json
from scipy.fft import fft, ifft

class PipelineTrack2D:
    def __init__(self,config,prefix):
        self.config = config
        self.prefix = prefix
        self.analpath = config['analpath']
        path = self.analpath+prefix+'/'+prefix+'_spots.csv'
        self.spots = pd.read_csv(path)
        self.savepath = '-tracked.'.join(path.split('.'))
        
    def link(self,search_range=3,memory=5,filter=False,points=None,min_length=10):
        """regions is a list of square ROIs represented as tuples 
           with ordering (xcenter,ycenter,halfwidth)"""
           
        spots = self.spots.dropna(subset=['x','y','frame'])

        if filter:
            #spots = spots = spots.loc[(spots['N0'] > 10) & (spots['N0'] < 1e4) & (spots['x_mle'] > 0) & (spots['y_mle'] > 0)]
            spots = tp.link_df(spots,search_range=search_range,memory=memory)
            spots = tp.filter_stubs(spots,min_length)
            spots = spots.reset_index(drop=True)

        else:
            spots = tp.link_df(spots, search_range=search_range, memory=memory)
            spots = spots.reset_index(drop=True)
            
        if points is not None:
            spots = self.add_dnearest(spots,points)

        return spots
        
    def add_dnearest(self,spots,points):
        """regions is a 2d array of circular ROIs represented as rows 
           with ordering (xcenter,ycenter)"""
           
        def is_within_region(x, y):
            distances = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
            return np.min(distances)
          
        centroids = spots[['x_mle','y_mle','particle']].groupby('particle').mean().reset_index()
        centroids['dnearest'] = centroids.apply(lambda row: is_within_region(row['x_mle'], row['y_mle']), axis=1)
        particles = centroids['particle'].unique()
        spots['dnearest'] = None
        for particle in particles:
            dnearest = centroids.loc[centroids['particle'] == particle,'dnearest'].values[0]
            spots.loc[spots['particle'] == particle,'dnearest'] = dnearest

        return spots
        
    def imsd(self,spots,mpp=0.1083,fps=10.0,max_lagtime=10):
        return tp.imsd(spots,mpp,fps,max_lagtime=max_lagtime,statistic='msd',pos_columns=['x_mle','y_mle'])
        
    def vanhove(self,spots,mpp=0.1083,fps=10.0,lagtime=10,bins=24):
        pos = spots.set_index(['frame', 'particle'])['x_mle'].unstack() # particles as columns
        vh = tp.vanhove(pos,lagtime,mpp=mpp,ensemble=True,bins=bins)
        return vh
        
    def vacf(self,spots,dt=1.0):
        """right now this is hard to use due to missed detections as NaNs"""
        pos = spots.set_index(['frame', 'particle'])['x_mle'].unstack() # particles as columns
        pos = pos.dropna(axis=1)
        V = np.diff(pos.values,axis=0)/dt
        F = fft(V,axis=0)
        return np.abs(F.conj()*F)
        #D = np.real(ifft(F.conj()*F,axis=0)) #autocorrelation
        #return D
                
    def save(self,spots):
        spots.to_csv(self.savepath)
        
        


