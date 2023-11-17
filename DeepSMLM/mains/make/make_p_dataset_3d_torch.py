import numpy as np
import uuid
import json
from SMLM.generators import Mix3D
from skimage.io import imsave

with open('make_p_dataset_3d_torch.json', 'r') as f:
    config = json.load(f)
    

N0space = config['N0']

for m in range(len(N0space)):

    this_config = config.copy()
    prefix_ = this_config['prefix'] + str(N0space[m])
    this_config['N0'] = N0space[m]
    this_config['prefix'] = prefix_
    stack = []; theta = []; spikes = []
    
    print(f'Generating set {prefix_}')
    for n in range(config['ngenerate']):
    
        print(f'Generating frame {n}')
        mix3d = Mix3D(this_config)
        adu,_spikes,_theta = mix3d.generate(margin=25)
        stack.append(adu); theta.append(_theta); spikes.append(_spikes.numpy())
        
    stack = np.array(stack); theta = np.array(theta); spikes = np.array(spikes)
    imsave(config['savepath']+prefix_+'_adu.tif',stack)
    imsave(config['savepath']+prefix_+'_spikes.tif',spikes)
    file = config['savepath']+prefix_+'_adu.npz'
    np.savez(file,theta=theta)
    
    del stack; del theta; del spikes
