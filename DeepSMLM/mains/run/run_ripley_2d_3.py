import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from SMLM.utils import ripley
from matplotlib import cm

with open('run_ripley_2d_3.json', 'r') as f:
    config = json.load(f)

prefixes = [
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_0min_5p-1_5',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_3min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_6min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_10min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_15min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_20min_5p-1_2'
]


prefixes = [
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_0min_5p-1_13',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_3min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_6min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_10min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_15min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_20min_5p-1_10'
]

prefixes = [
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_0min_5p-1_3',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_3min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_6min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_10min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_15min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_20min_5p-1_1'
]

prefixes = [
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_0min_control_2',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_3min_control_2',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_6min_control_2',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_10min_control_2',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_15min_control_2',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_20min_control_2'
]


viridis = cm.get_cmap('viridis')
colors = viridis(np.linspace(0, 1, len(prefixes)))

radii = np.linspace(0,2,5)
fig,ax=plt.subplots()
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    print(len(spots))
    #spots = spots.loc[spots['peak'] > 200]
    #spots = spots.sample(4000)
    K,radii = ripley(spots,sample_size=(10,10),radii=radii,correct=False,plot_gmm=False)
    ax.plot(radii,K,color=colors[n],label=prefix)
    print(prefix,K)
ax.legend()
plt.show()
