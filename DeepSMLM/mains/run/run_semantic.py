from pipes import PipelineCount2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'230816_Hela_JQ1_25uM_2h_1_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_2_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_3_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_4_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_5_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_6_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_7_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_8_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_9_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_2h_10_MMStack_Default.ome'
]


with open('run_semantic.json', 'r') as f:
    config = json.load(f)
    
for prefix in prefixes:
    print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.count()
    del pipe
