import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from SMLM.utils import ripley

with open('run_ripley_2d.json', 'r') as f:
    config = json.load(f)

prefixes1 = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_5',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_7',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_8',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_9',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_11',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_19',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_21',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_25',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_28',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_29'
]

prefixes2 = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_6',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_9',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_11',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_12',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_13',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_14',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_15',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_16',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_17',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_20'
]

prefixes3 = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_1',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_2',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_3',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_5',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_7',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_8',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_12',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_13',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_17',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_18'
]

prefixes4 = [
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4',
'230726_U2OS_J549_10pM_1hr_j646_10pM_overnight_5mW_20mW-R-4_1-sub',
'230830_Hela_j646_10pM_overnight_1000frames_20mW_Control-3',
'230830_Hela_j646_10pM_overnight_1000frames_20mW_Control-7'
]

radii = np.linspace(0,2,5)
Kavg = []
for n,prefix in enumerate(prefixes1):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['peak'] > 200]
    spots = spots.sample(10000)
    K,radii = ripley(spots,sample_size=(10,10),radii=radii,correct=False,plot_gmm=False)
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
    Kavg.append(K)
Kavg = np.array(Kavg)
Kavg = np.mean(Kavg,axis=0)
plt.plot(radii,Kavg,color='blue')

Kavg = []
for n,prefix in enumerate(prefixes3):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    spots = spots.loc[spots['peak'] > 200]
    spots = spots.sample(10000)
    K,radii = ripley(spots,sample_size=(10,10),radii=radii,correct=False,plot_gmm=False)
    Kavg.append(K)
Kavg = np.array(Kavg)
Kavg = np.mean(Kavg,axis=0)   
plt.plot(radii,Kavg,color='black')


plt.show()
