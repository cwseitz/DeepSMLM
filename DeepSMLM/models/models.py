import torch
import os
import json
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationCNN
from DeepSMLM.torch.train.metrics import jaccard_coeff
from DeepSMLM.localize import NeuralEstimator2D, NeuralEstimatorLoG2D

class Ring_Rad1_K5_CMOS:
    def __init__(self,ppconfig):
        self.modelpath = os.path.dirname(os.path.realpath(__file__)) + '/'
        model_config = {'modelpath':self.modelpath,
                        'modelname':'ring_rad1_k5_cmos'
                       }
        self.config = {**ppconfig,**model_config}
        
    def forward(self,adu):
        self.estimator = NeuralEstimator2D(self.config)
        spots,outputs = self.estimator.forward(adu)
        return spots,outputs

class Ring_Rad1_K5_SPAD:
    def __init__(self,ppconfig):
        self.modelpath = os.path.dirname(os.path.realpath(__file__)) + '/'
        model_config = {'modelpath':self.modelpath,
                        'modelname':'ring_rad1_k5_spad'
                       }
        self.config = {**ppconfig,**model_config}
        
    def forward(self,adu):
        self.estimator = NeuralEstimator2D(self.config)
        spots,outputs = self.estimator.forward(adu)
        return spots,outputs

