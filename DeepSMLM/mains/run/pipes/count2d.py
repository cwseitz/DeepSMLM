from SMLM.utils.tile import *
from SMLM.segment import SegmentCNN, SegmentThreshold
from SMLM.utils.count import CountMatrix
from skimage.io import imsave
from pathlib import Path
from pycromanager import Dataset
from SMLM.localize import LoGDetector
from skimage.measure import label
import napari
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt

class PipelineCount2D:
    """Pipeline for counting spots within segmented objects"""
    def __init__(self,config,prefix,ndtiff=False):
        self.config = config
        self.prefix = prefix
        Path(self.config['analpath']+self.prefix).mkdir(parents=True, exist_ok=True)

        if ndtiff: #this isn't integrated
            dataset = Dataset(self.config['datapath']+self.prefix)
            self.X = dataset.as_array(stitched=False,axes=['z','channel','row','column'])
            nz,nc,nt,_,nx,ny = self.X.shape
            self.X = self.rawdata.reshape((nz,nc,nt**2,nx,ny))
            self.X = self.X[:,:,:,:1844,:1844]
        else:
            self.X = tifffile.imread(self.config['datapath']+self.prefix+'.tif')

    def segment(self):
        ns = SegmentThreshold(self.X,self.config['threshold'],self.config['filters'])
        mask = ns.segment(plot=True)
        
    def localize(self,plot=False):
        path = self.config['analpath']+self.prefix+'/'+self.prefix+'_spots.csv'
        file = Path(path)
        threshold = self.config['thresh_log']
        log = LoGDetector(self.X,threshold=threshold)
        spots = log.detect() #image coordinates
        if plot:
            log.show(); plt.show()
        spots.to_csv(path)
    
    def count(self):
        path = self.config['analpath']+self.prefix+'/'+self.prefix+'_spots.csv'
        spots = pd.read_csv(path)
        mask = self.config['analpath']+self.prefix+'-mask.tif'
        mask = label(tifffile.imread(mask))
        cm = CountMatrix(spots,mask)   
        counts = cm.get_counts()
        return counts


