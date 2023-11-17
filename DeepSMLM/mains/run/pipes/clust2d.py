import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.stats import kde
from sklearn.cluster import DBSCAN

class PipelineCluster2D:
    def __init__(self,config,prefix,tmax=1000):
        self.config = config
        self.prefix = prefix
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        Path(self.analpath+self.prefix).mkdir(parents=True, exist_ok=True)
        self.spots_base = self.load_dataset()
        self.spots_base = self.spots_base.loc[self.spots_base['frame'] < tmax]

    def load_dataset(self):
        path = self.config['datapath']+'/'+self.prefix
        mask = tifffile.imread(path+'-mask.tif')
        path = self.config['analpath']+self.prefix+'/'+self.prefix
        spots = pd.read_csv(path+'_spots.csv')
        spots = spots.dropna()
        spots['x'] = spots['x'].astype(int)
        spots['y'] = spots['y'].astype(int)
        spots['mask_value'] = mask[spots['x'],spots['y']]
        spots = spots[spots['mask_value'] > 0]
        return spots

    def getROI(self,spots,center,hw=20):
        xr,yr = center
        spotsROI = spots.loc[(spots['x_mle'] > xr-hw) & 
                             (spots['x_mle'] < xr+hw) & 
                             (spots['y_mle'] > yr-hw) & 
                             (spots['y_mle'] < yr+hw)]                
        return spotsROI    
        
    def add_interval(self,spots,interval_size=10):
        max_frame = spots['frame'].max()
        intervals = range(0, max_frame + interval_size, interval_size)
        spots['interval'] = pd.cut(spots['frame'], bins=intervals, right=False, labels=intervals[:-1])
        return spots

    def clustDBSCAN(self,spots):
        X = spots[['x_mle','y_mle']].values
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X)
        spots['cluster'] = clusters
        return spots

    def save(self,spots,suffix):
        path = self.analpath+self.prefix+'/'+self.prefix+f'_spots_{suffix}.csv'
        spots.to_csv(path)

    def showKDE(self,spots,ax,nbins=100,hw=15):
        x, y = spots['x_mle'], spots['y_mle']
        k = kde.gaussian_kde(spots[['x_mle','y_mle']].to_numpy().T)
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))    
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='plasma')
        

    def scatter(self,spots,ax,scalebar_x=140.0,scalebar_y=10.0,scalebar_length=30.0):
        ax.invert_yaxis() #for top-left origin
        splot = ax.scatter(spots['y_mle'],spots['x_mle'],color='black',marker='x',s=1)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()

    def scatter_colored(self,spots,column,ax,cmap='plasma',scalebar_x=140.0,scalebar_y=10.0,scalebar_length=30.0,colorbar=False,clabel='Time (s)'):
        ax.invert_yaxis() #for top-left origin
        splot = ax.scatter(spots['y_mle'],spots['x_mle'],
                   c=spots[column], cmap=cmap, 
                   marker='x',s=1)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        if colorbar:
            bar = plt.colorbar(splot)
            bar.set_label(clabel)
        plt.tight_layout()



