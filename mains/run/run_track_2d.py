import json
from pipes import *
import pandas as pd
from SMLM.utils import make_animation
from skimage.io import imread

prefixes = [
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__2',
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__4',
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__5',
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__7',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__12',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__15',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__19',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__31',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__4'
]


with open('run_track_2d.json', 'r') as f:
    config = json.load(f)
    config['analpath'] += 'BRD4/'
    config['datapath'] += 'BRD4/'

all_spots = []
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    pipe = PipelineTrack2D(config,prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    stack = imread(config['datapath']+prefix+'.tif')
    make_animation(stack,spots)
    spots = pipe.link(filter=True,min_length=10,search_range=1)
    spots['particle'] += 20000*n #just pick a big number (larger than the number of particles per cell)
    all_spots.append(spots)
    
pipe = PipelineTrack2D(config,prefix)
all_spots = pd.concat(all_spots)
im_brd4 = pipe.imsd(all_spots,max_lagtime=5)


###############################################

prefixes = [
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__1',
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__6',
'231009_Cotransfection_BRD4 1000ng_H2B 1000ng_5pm_1hours_L640-5mW_L488-10mW_100ms__7',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-10mW_L488-10mW_100ms__2',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__11',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__12',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__13',
'231009_Cotransfection_BRD4 1000ng_H2B 500ng_5pm_1hours_L640-5mW_L488-10mW_100ms__5'
]

with open('run_track_2d.json', 'r') as f:
    config = json.load(f)
    config['analpath'] += 'H2B/'
    config['datapath'] += 'H2B/'

all_spots = []
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    pipe = PipelineTrack2D(config,prefix)
    #spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    #stack = imread(config['datapath']+prefix+'.tif')
    #make_animation(stack,spots)
    spots = pipe.link(filter=True,min_length=10,search_range=1)
    spots['particle'] += 20000*n #just pick a big number (larger than the number of particles per cell)
    all_spots.append(spots)

pipe = PipelineTrack2D(config,prefix)
all_spots = pd.concat(all_spots)
im_h2b = pipe.imsd(all_spots,max_lagtime=5)

###############################################

fig, ax = plt.subplots()
avg_msd_brd4 = np.mean(im_brd4.values,axis=1)
avg_msd_brd4 = np.insert(avg_msd_brd4,0,0)
avg_msd_h2b = np.mean(im_h2b.values,axis=1)
avg_msd_h2b = np.insert(avg_msd_h2b,0,0)
ax.plot(avg_msd_brd4,color='blue',label='BRD4')
ax.plot(avg_msd_h2b,color='red',label='H2B')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
       xlabel='lag time $t$')
#ax.set_xscale('log')
#ax.set_yscale('log')

plt.show()


