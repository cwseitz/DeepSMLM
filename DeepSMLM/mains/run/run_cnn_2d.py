from SMLM.utils.pipes import *
from SMLM.torch.dataset import SMLMDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-1'
]

with open('run_storm_cnn_2d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineCNN2D(config,dataset)
    spots = pipe.localize()
    pipe.plot(spots)


