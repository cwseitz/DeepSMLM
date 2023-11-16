import json
from pipes import *
import pandas as pd

prefixes = [
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__1'
]

with open('run_track_2d_roi.json', 'r') as f:
    config = json.load(f)

d = 5.0
for prefix in prefixes:
    print("Processing " + prefix)
    spotsROI = pd.read_csv(config['analpath2']+prefix+'/'+prefix+'_spots.csv')
    points = spotsROI[['x','y']].values
    pipe = PipelineTrack2D(config,prefix)
    spots = pipe.link(filter=True,points=points)
    im = pipe.imsd(spots,max_lagtime=50).values
    tspots = spots[['particle','dnearest']].groupby('particle').mean().reset_index()
    isnear = (tspots['dnearest'] < d).values #this should be sorted correctly
    nt,ntraj = im.shape
    fig, ax = plt.subplots()
    for n in range(ntraj):
        color = 'black'
        if isnear[n]: color = 'red'
        ax.plot(im[:,n],color=color,alpha=0.1)
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
           xlabel='lag time $t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    #pipe.save(spots)
