import numpy as np
import json
import matplotlib.pyplot as plt
from SMLM.generators import Mix2D_SR3_Ring
from skimage.io import imsave
from scipy.ndimage import zoom

with open('make_sr3_dataset_2d.json', 'r') as f:
    config = json.load(f)
    
kspace = config['particles']

for m in range(len(kspace)):

    this_config = config.copy()
    prefix_ = this_config['prefix'] + str(kspace[m])
    this_config['particles'] = kspace[m]
    this_config['prefix'] = prefix_
    adu_h_all = []; adu_l_all = []; adu_sr_all = []
    
    print(f'Generating set {prefix_}')
    for n in range(config['ngenerate']):
    
        print(f'Generating frame {n}')
        mix2d = Mix2D_SR3_Ring(this_config)
        adu_h,adu_l = mix2d.generate(plot=False)
        adu_l = np.squeeze(adu_l)
        zoom_factors = (8,8)
        adu_sr = zoom(adu_l, zoom_factors, order=3)
        adu_h_all.append(adu_h); adu_l_all.append(adu_l); adu_sr_all.append(adu_sr)
                
    adu_h_all = np.array(adu_h_all); adu_l_all = np.array(adu_l_all)
    adu_sr_all = np.array(adu_sr_all)

    imsave(config['savepath']+prefix_+'_adu_hr.tif',adu_h_all)
    imsave(config['savepath']+prefix_+'_adu_lr.tif',adu_l_all)
    imsave(config['savepath']+prefix_+'_adu_sr.tif',adu_sr_all)
    
    del adu_h_all; del adu_l_all; del adu_sr
