import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from SMLM.utils import ripley

with open('run_ripley_2d_2.json', 'r') as f:
    config = json.load(f)

prefixes1 = [
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-1',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-3',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-5',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-8',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-9'
]

prefixes2 = [
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-1',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-3',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-4',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-8'
]

radii = np.linspace(0,2,5)
Kavg = []
for n,prefix in enumerate(prefixes1):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['peak'] > 200]
    spots = spots.sample(10000)
    K,radii = ripley(spots,sample_size=(10,10),radii=radii,correct=False,plot_gmm=False)
    plt.plot(radii,K,color='red')
    Kavg.append(K)
Kavg = np.array(Kavg)
Kavg = np.mean(Kavg,axis=0)
plt.plot(radii,Kavg,color='red')
    
Kavg = []
for n,prefix in enumerate(prefixes2):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['peak'] > 200]
    spots = spots.sample(10000)
    K,radii = ripley(spots,sample_size=(10,10),radii=radii,correct=False,plot_gmm=False)
    plt.plot(radii,K,color='blue')
    Kavg.append(K)
Kavg = np.array(Kavg)
Kavg = np.mean(Kavg,axis=0)
plt.plot(radii,Kavg,color='blue')

plt.show()
