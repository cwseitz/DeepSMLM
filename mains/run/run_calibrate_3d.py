from SMLM.utils.pipes import *
from SMLM.torch.dataset import SMLMDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'230722_QD-Cal-HighNA-Astigmatism-crop-substack'
]

with open('run_calibrate_3d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineCalibrate3D(config,dataset)
    pipe.calibrate(10,11)

    


