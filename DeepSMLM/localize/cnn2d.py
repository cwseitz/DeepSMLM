import argparse
import collections
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from DeepSMLM.torch.utils import prepare_device
from DeepSMLM.torch.models import LocalizationCNN
from torch.nn import Module, MaxPool2d, ConstantPad3d
from torch.nn.functional import conv3d

def tensor_to_np(x):
    return np.squeeze(x.cpu().detach().numpy())

class NeuralEstimator2D:
    def __init__(self,config,pixel_size=108.3):
        self.modelpath = config['modelpath']
        self.modelname = config['modelname']
        self.model,self.device = self.load_model()
        self.spots = pd.DataFrame(columns=['x','y'])
        self.pprocessor = PostProcessor2D(config['thresh_cnn'],config['radius'],pixel_size=pixel_size,device=self.device)
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
        xyb = []
        for n in range(nb):
            print(f'Model forward on frame {n}')
            output = self.model(stack[n].unsqueeze(0))
            print(output.shape)
            xy = self.pprocessor.forward(output)
            spots = pd.DataFrame(xy,columns=['x','y'])
            spots = spots.assign(frame=n)
            xyb.append(spots)
        return pd.concat(xyb,ignore_index=True)
        
class PostProcessor2D(Module):
    def __init__(self,threshold,radius,pixel_size=108.3,device='cpu'):
        super().__init__()
        self.device = device
        self.thresh = threshold
        self.r = radius
        self.pixel_size_lateral = pixel_size
        self.maxpool = MaxPool2d(kernel_size=2*self.r + 1, stride=1, padding=self.r)
        self.zero = torch.FloatTensor([0.0]).to(self.device)

    def get_conf_vol(self,pred_vol):
        pred_thresh = torch.where(pred_vol > self.thresh, pred_vol, self.zero)
        conf_vol = self.maxpool(pred_thresh)
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)
        conf_vol = torch.squeeze(conf_vol)
        return conf_vol
 
    def forward(self,pred_vol,plot=False):                
        conf_vol = self.get_conf_vol(pred_vol)
        if plot:
            fig,ax=plt.subplots(1,2)
            ax[0].imshow(pred_vol[0,0].detach().numpy())
            ax[1].imshow(conf_vol.detach().numpy())
            plt.show()
        batch_idx = torch.nonzero(conf_vol)
        xbool, ybool = batch_idx[:, 0], batch_idx[:, 1]
        xbool, ybool = tensor_to_np(xbool), tensor_to_np(ybool)
        xrec = xbool/4; yrec = ybool/4
        xyz_rec = np.column_stack((xrec, yrec)) + 0.5
        return xyz_rec

        
        
        
        

