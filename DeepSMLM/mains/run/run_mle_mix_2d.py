from pipes import PipelineMLE2D
from SMLM.torch.dataset import SMLMDataset
from SMLM.psf.psf2d.mix import MLE2DMix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


prefixes = [
'230720_Mix2D_k5_adu'
]

with open('run_mle_mix_2d.json', 'r') as f:
    config = json.load(f)

margin=5
for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    nt,ntheta,nspots = dataset.theta.shape
    for n in range(nt):
        theta0 = np.zeros((nspots,ntheta))
        nx,ny = dataset.stack[n,0].shape
        xinit = np.random.uniform(margin,nx-margin,nspots)
        yinit = np.random.uniform(margin,ny-margin,nspots)
        theta0[:,0] = xinit; theta0[:,1] = yinit
        theta0[:,2] = 0.92; theta0[:,3] = 300
        opt = MLE2DMix(theta0,dataset.stack[n],config)
        opt.optimize()
