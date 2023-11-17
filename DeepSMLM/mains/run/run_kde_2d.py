import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from SMLM.utils import KDE
from tifffile import imread
from skimage.io import imsave

with open('run_kde_2d.json', 'r') as f:
    config = json.load(f)

prefixes = [
'231021_control_H2B 1000ng_8pm_overnight_L640-20mW_10ms__19',
'231021_control_H2B 1000ng_8pm_overnight_L640-20mW_10ms__20',
'231021_control_H2B 1000ng_8pm_overnight_L640-20mW_10ms__2',
'231021_control_H2B 1000ng_8pm_overnight_L640-20mW_10ms__3',
'231021_control_H2B 1000ng_8pm_overnight_L640-20mW_10ms__7'
]
plot=False
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[(spots['N0'] > 10) & (spots['N0'] < 5000) & (spots['x_mle'] > 0) & (spots['y_mle'] > 0)]
    spots = spots.sample(10000)
    render = KDE(spots).get_kde(sigma=2.0) #in high-res pixel units (10nm by default)
    
    if plot:
        fig,ax=plt.subplots()
        ax.imshow(render,cmap='gray',vmin=0,vmax=0.2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        fig,ax=plt.subplots()
        ax.scatter(spots['y_mle'],spots['x_mle'],s=1,color='black',marker='x')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        plt.show()
    
    imsave(config['savepath']+prefix+'-kde.tif',render)
