import argparse
import collections
import torch
import numpy as np
import json
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationCNN
from torch.nn import Module, MaxPool3d, ConstantPad3d
from torch.nn.functional import conv3d

class NeuralEstimator3D:
    def __init__(self,config):
        self.config = config
        self.modelpath = config['modelpath']
        self.modelname = config['modelname']
        self.model,self.device = self.load_model() 
        self.pprocessor = PostProcessor3D(config,device=self.device)
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(self.modelpath+self.modelname+'/config.json', 'r') as f:
            train_config = json.load(f)
        args = train_config['arch']['args']
        model = LocalizationCNN(args['nz'],args['scaling_factor'],dilation_flag=args['dilation_flag'])
        model = model.to(device=device)
        checkpoint = torch.load(self.modelpath+self.modelname+'/'+self.modelname+'.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device
    def forward(self,stack,show=False):
        stack = stack.astype(np.float32)
        stack = torch.from_numpy(stack)
        stack = stack.to(self.device)
        nb,nc,nx,ny = stack.shape
        xyzb = []
        for n in range(nb):
            print(f'Model forward on frame {n}')
            output = self.model(stack[n].unsqueeze(0))
            xyz = self.pprocessor.forward(output) #very slow
            spots = pd.DataFrame(xyz,columns=['x','y','z'])
            spots = spots.assign(frame=n)
            xyzb.append(spots)
        return pd.concat(xyzb,ignore_index=True)
        
def tensor_to_np(x):
    return np.squeeze(x.cpu().detach().numpy())

class PostProcessor3D(Module):
    def __init__(self,config,device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.thresh = config['thresh_cnn']
        self.r = config['radius']
        self.pixel_size_axial = 2*config['zhrange']/config['nz']
        self.upsampling_shift = 0
        self.maxpool = MaxPool3d(kernel_size=2*self.r + 1, stride=1, padding=self.r)
        self.pad = ConstantPad3d(self.r, 0.0)
        self.zero = torch.FloatTensor([0.0]).to(self.device)

    def get_conf_vol(self,pred_vol):
        pred_thresh = torch.where(pred_vol > self.thresh, pred_vol, self.zero)
        conf_vol = self.maxpool(pred_thresh)
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)
        conf_vol = torch.squeeze(conf_vol)
        return conf_vol
 
    def forward(self,pred_vol):
        num_dims = len(pred_vol.size())
        if np.not_equal(num_dims, 5):
            if num_dims == 4:
                pred_vol = pred_vol.unsqueeze(0)
            else:
                pred_vol = pred_vol.unsqueeze(0)
                pred_vol = pred_vol.unsqueeze(0)
                
        conf_vol = self.get_conf_vol(pred_vol)
        batch_idx = torch.nonzero(conf_vol)
        zbool, xbool, ybool = batch_idx[:, 0], batch_idx[:, 1], batch_idx[:, 2]
        xbool, ybool, zbool = tensor_to_np(xbool), tensor_to_np(ybool), tensor_to_np(zbool)
        xrec = xbool/4; yrec = ybool/4; zrec = zbool
        zrec = zrec - self.config['zhrange']/self.pixel_size_axial
        xyz_rec = np.column_stack((xrec, yrec, zrec)) + 0.5

        return xyz_rec
        
        
        
        

