import numpy as np
import pandas as pd
import tifffile
import napari
import torch
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from pycromanager import Dataset
from skimage.filters import gaussian, threshold_otsu
from skimage.io import imread, imsave
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage.util import img_as_ubyte, img_as_uint, map_array, img_as_bool
from skimage.measure import regionprops_table, label, find_contours
from scipy import ndimage as ndi
from UNET.torch_models import UNetModel
import torch.nn.functional as F

class SegmentCNN:
    def __init__(self,X,filters,modelpath,p0=0.95):
        self.X = X
        self.p0 = p0
        self.filters = filters
        self.modelpath = modelpath
                
    def sfmx_to_mask(self,sfmx):
        sfmx_interior = sfmx[0,1]
        mask = np.zeros((256,256),dtype=np.bool)
        mask[sfmx_interior >= self.p0] = True
        mask = img_as_bool(mask)
        mask = gaussian(mask,sigma=0.2)
        mask[mask > 0] = 1
        mask = img_as_bool(mask)
        return mask 

    def cnn(self,X):
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNetModel(1,3)
        model.to(device=device)
        checkpoint = torch.load(self.modelpath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        with torch.no_grad():
            X = X - X.mean()
            image = torch.from_numpy(X).unsqueeze(0).unsqueeze(0)
            image = image.to(device=device, dtype=torch.float)
            output = model(image).cpu()
            sfmx = F.softmax(output,dim=1)
                
        mask = self.sfmx_to_mask(sfmx)
        return sfmx, mask
        
    def segment(self,correct=True,plot=False):
        X = resize(self.X,(256,256))
        sfmx,mask = self.cnn(X)
        if plot:
            self.plot(sfmx); plt.tight_layout(); plt.show()
        mask = img_as_bool(resize(mask,self.X.shape))
        mask = self.filter_objects(mask,**self.filters)
        labeled = label(mask).astype(np.uint16)
        return labeled
    
    def filter_objects(self,mask,min_area=None,max_area=None,min_solid=None):
        mask = label(mask)
        props = ('label', 'area', 'solidity')
        table = regionprops_table(mask,properties=props)
        condition = (table['area'] > min_area) &\
                    (table['area'] < max_area) &\
                    (table['solidity'] > min_solid)
        input_labels = table['label']
        output_labels = input_labels * condition
        filtered_mask = map_array(mask,input_labels,output_labels)
        filtered_table = regionprops_table(filtered_mask,properties=props)
        table = pd.DataFrame(table)
        filtered_table = pd.DataFrame(filtered_table)
        filtered_mask = clear_border(filtered_mask)
        filtered_mask[filtered_mask > 0]  = 1
        return filtered_mask

    def plot(self,sfmx):
        fig,ax = plt.subplots(1,4,figsize=(12,2))
        im0 = ax[0].imshow(self.X,cmap='gray')
        ax[0].set_title('Raw')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        im1 = ax[1].imshow(sfmx[0,0,:,:],cmap='coolwarm')
        ax[1].set_title('Background')
        ax[1].set_xticks([]); ax[1].set_yticks([])
        plt.colorbar(im1,ax=ax[1],label='Probability')
        im2 = ax[2].imshow(sfmx[0,1,:,:],cmap='coolwarm')
        ax[2].set_title('Interior')
        ax[2].set_xticks([]); ax[1].set_yticks([])
        plt.colorbar(im2,ax=ax[2],label='Probability')
        im3 = ax[3].imshow(sfmx[0,2,:,:],cmap='coolwarm')
        ax[3].set_xticks([]); ax[2].set_yticks([])
        ax[3].set_title('Boundary')
        plt.colorbar(im3,ax=ax[3],label='Probability')
        plt.tight_layout()
        
class SegmentThreshold:
    def __init__(self,X,threshold,filters):
        self.X = X
        self.filters = filters
        self.threshold = threshold

    def thresh(self,X):
        binary = X > self.threshold
        return binary
    
    def segment(self,plot=False):
        self.X = self.X/self.X.max()
        mask = self.thresh(self.X)
        if plot:
            self.plot(mask); plt.tight_layout(); plt.show()
        mask = self.filter_objects(mask,**self.filters)
        labeled = label(mask).astype(np.uint16)
        return labeled
    
    def filter_objects(self,mask,min_area=None,max_area=None,min_solid=None):
        mask = label(mask)
        props = ('label', 'area', 'solidity')
        table = regionprops_table(mask,properties=props)
        condition = (table['area'] > min_area) &\
                    (table['area'] < max_area) &\
                    (table['solidity'] > min_solid)
        input_labels = table['label']
        output_labels = input_labels * condition
        filtered_mask = map_array(mask,input_labels,output_labels)
        filtered_table = regionprops_table(filtered_mask,properties=props)
        table = pd.DataFrame(table)
        filtered_table = pd.DataFrame(filtered_table)
        filtered_mask = clear_border(filtered_mask)
        filtered_mask[filtered_mask > 0]  = 1
        return filtered_mask
        
    def plot(self,mask):
        fig,ax = plt.subplots(1,2,figsize=(6,2))
        im0 = ax[0].imshow(self.X,cmap='gray')
        ax[0].set_title('Raw')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        im1 = ax[1].imshow(mask,cmap='coolwarm')
        ax[1].set_title('Mask')
        ax[1].set_xticks([]); ax[1].set_yticks([])
        plt.tight_layout()
