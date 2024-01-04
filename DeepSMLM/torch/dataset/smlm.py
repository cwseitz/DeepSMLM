import numpy as np
import tifffile
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from glob import glob

class SMLMDataset(Dataset):
    def __init__(self,path,name):
        self.stack = tifffile.imread(path+'.tif')
        self.name = name
        try:
            prefix = path.split('.')[0]
            npz = np.load(prefix+'.npz',allow_pickle=True)
            self.theta = npz['theta']
        except Exception as e:
            self.theta = None
    def get_theta(self,idx):
        if self.theta is not None:
            return self.theta[idx]
        else:
            return None
    def __len__(self):
        nb,nc,nx,ny = self.stack.shape
        return nb
    def __getitem__(self, idx):
        adu = self.stack[idx]
        return adu

class SMLMDataset_Train(Dataset):
    """Looks for individal files for training"""
    def __init__(self,path,name):
        self.files_input = sorted(glob(path+'*adu*.tif'))
        self.files_target = sorted(glob(path+'*spikes*.tif'))
    def __len__(self):
        return len(self.files_input)
    def __getitem__(self, idx):
        input = imread(self.files_input[idx]).astype(np.float32)
        target = imread(self.files_target[idx])
        input = np.expand_dims(input,0)
        target = np.expand_dims(target,0)
        return input,target
        
class VAEDataset_Train(Dataset):
    """Looks for individual files for training"""
    def __init__(self,path,name):
        self.files_input = sorted(glob(path+'*adu*.tif'))
        self.files_target = sorted(glob(path+'*sum*.tif'))
        #self.file_target = np.load(glob(path+'*.npz')[0])['theta']
        #self.file_target = self.file_target[:,:2,:]
        #nsamples,nparams,nspots = self.file_target.shape
        #self.file_target = self.file_target.reshape((nsamples,nparams*nspots))
    def __len__(self):
        return len(self.files_input)
    def __getitem__(self, idx):
        input = imread(self.files_input[idx]).astype(np.float32)
        target = imread(self.files_target[idx]).astype(np.float32)
        target = target/target.max()
        #target = self.files_target[idx]
        input = np.expand_dims(input,0)
        target = np.expand_dims(target,0)
        return input,target

