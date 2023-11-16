from pipes import PipelineCluster2D
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4'
]

with open('run_storm_clust_2d.json', 'r') as f:
    config = json.load(f)

ROI = [(130,145),(150,180),(90,175)]    
for prefix in prefixes:
    print("Processing " + prefix)
    pipe = PipelineCluster2D(config,prefix,tmax=500)
    fig,ax=plt.subplots()
    pipe.scatter(pipe.spots_base,ax)
    plt.show()

    for n,roi in enumerate(ROI):
        fig,ax=plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=True)  
        spots_roi = pipe.getROI(pipe.spots_base,roi,hw=50)
        spots_roi = pipe.clustDBSCAN(spots_roi)
        spots_roi = pipe.add_interval(spots_roi)
        pipe.save(spots_roi,f'clustered_ROI{n}')
        
        pipe.scatter(spots_roi,ax[0])
        spots_roi['interval'] = pd.to_numeric(spots_roi['interval'])
        spots_roi['interval'] = spots_roi['interval']*0.01
        pipe.scatter_colored(spots_roi,'interval',ax[1],colorbar=True,cmap='coolwarm')
        spots_roi = spots_roi.loc[spots_roi['cluster'] > -1]
        pipe.scatter_colored(spots_roi,'cluster',ax[2],cmap='rainbow')
        plt.tight_layout()
        plt.show()





