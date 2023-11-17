import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from SMLM.utils import make_animation
from tifffile import imread

with open('run_animate_2d.json', 'r') as f:
    config = json.load(f)

prefixes = [
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__1',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__2',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__27',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__28',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__31',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__32',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__33',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__6'
]

for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    stack = imread(config['datapath']+prefix+'.tif')
    make_animation(stack,spots)
    plt.show()
