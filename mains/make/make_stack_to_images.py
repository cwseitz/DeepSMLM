import tifffile
import numpy as np
import json
from skimage.io import imsave
from glob import glob

""""Decompose datasets for training"""

def stack_to_images(stack,base_name):
    nb = stack.shape[0]
    for n in range(nb):
        imsave(base_name+f'-{n}.tif',np.squeeze(stack[n]))

with open('make_stack_to_images.json', 'r') as f:
    config = json.load(f)

stacks = sorted(glob(config['path']+'*_adu.tif'))
for stack_name in stacks:
    base_name = stack_name.split('.')[0].split('/')[-1]
    stack = tifffile.imread(stack_name)
    stack_to_images(stack,config['savepath']+base_name)

spikes = False
if spikes:
    stacks = sorted(glob(config['path']+'*_spikes.tif'))
    for stack_name in stacks:
        base_name = stack_name.split('.')[0].split('/')[-1]
        stack = tifffile.imread(stack_name)
        stack_to_images(stack,config['savepath']+base_name)
    

