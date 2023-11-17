from pipes import PipelineMLE2D
from SMLM.torch.dataset import SMLMDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
#'230923_j646_5pM_1.5hours_200frames_10mW_live__1',
#'230923_j646_5pM_1.5hours_200frames_10mW_live__8',
#'230923_j646_5pM_1.5hours_200frames_10mW_live__9'
'230929_Hela_H2B_1000ng_8h_200frames_10mW_100ms_J646_10PM_1.25hours_fixed__8',
'230929_Hela_H2B_1000ng_8h_200frames_10mW_100ms_J646_10PM_1.25hours_fixed__10',
'230929_Hela_H2B_1000ng_8h_200frames_10mW_100ms_J646_10PM_1.25hours_fixed__11'
]

prefixes = [
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-20mW_10ms__4',
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__1',
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__22',
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__23',
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__2',
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__31',
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__32',
#'231028_CONTROL_H2B 1000ng_7pm_15h__L640-30mW_10ms__7',
#'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__11',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__1',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__27',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__28',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__2',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__31',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__32',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__33',
'231028_JQ1_1uM_8hours_H2B 1000ng_7pm_15h__L640-30mW_10ms__6'
]




with open('run_mle_2d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineMLE2D(config,dataset)
    pipe.localize(plot_spots=False,plot_fit=False,fit=True,tmax=2000,run_deconv=False)
