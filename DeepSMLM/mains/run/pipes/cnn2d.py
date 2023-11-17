import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from SMLM.localize import NeuralEstimator2D


class PipelineCNN2D:
    def __init__(self,config,dataset):
        self.config = config
        self.dataset = dataset
        self.estimator = NeuralEstimator2D(config)
    def localize(self):
        spots = self.estimator.forward(self.dataset.stack)
        return spots
        
    def plot(self,spots):
        nb,nc,nx,ny = self.dataset.stack.shape
        for n in range(nb):
            _spots = spots.loc[spots['frame'] == n]
            _theta = self.dataset.theta[n]
            fig,ax=plt.subplots()
            ax.imshow(self.dataset.stack[n,0],cmap='gray')
            ax.invert_yaxis()
            ax.scatter(_spots['y'],_spots['x'],marker='x',color='red',s=8)
            ax.scatter(_theta[1],_theta[0],marker='x',color='blue',s=8)
            plt.show()
        




