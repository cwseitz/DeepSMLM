from pipes import PipelineCount2D
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

with open('run_count_2d.json', 'r') as f:
    config = json.load(f)

prefixes_0h = [
'230816_Hela_JQ1_40uM_0h_1_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_2_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_3_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_4_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_5_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_6_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_7_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_9_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_0h_10_MMStack_Default.ome'
]

counts_0h = []
for prefix in prefixes_0h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    counts_0h.append(count_mat)
    del pipe

path = config['analpath']+'230816_Hela_JQ1_40uM_0h_counts.csv'
counts_0h = pd.concat(counts_0h)
counts_0h.to_csv(path)

prefixes_2h = [
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

counts_2h = []
for prefix in prefixes_2h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    #counts_2h += count_mat['counts'].tolist()
    counts_2h.append(count_mat)
    del pipe
    
path = config['analpath']+'230816_Hela_JQ1_25uM_2h_counts.csv'
counts_2h = pd.concat(counts_2h)
counts_2h.to_csv(path)

prefixes_4h = [
#'230816_Hela_JQ1_25uM_4h_1_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_2_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_3_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_4_MMStack_Default.ome',
#'230816_Hela_JQ1_25uM_4h_5_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_6_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_7_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_8_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_12_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_4h_13_MMStack_Default.ome'
]

counts_4h = []
for prefix in prefixes_4h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    counts_4h.append(count_mat)
    del pipe

path = config['analpath']+'230816_Hela_JQ1_25uM_4h_counts.csv'
counts_4h = pd.concat(counts_4h)
counts_4h.to_csv(path)


prefixes_6h = [
'230816_Hela_JQ1_25uM_6h_1_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_2_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_3_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_4_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_5_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_6_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_7_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_8_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_9_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_6h_10_MMStack_Default.ome'
]

counts_6h = []
for prefix in prefixes_6h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    counts_6h.append(count_mat)
    del pipe

path = config['analpath']+'230816_Hela_JQ1_25uM_6h_counts.csv'
counts_6h = pd.concat(counts_6h)
counts_6h.to_csv(path)


prefixes_9h = [
'230816_Hela_JQ1_25uM_9h_1_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_2_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_3_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_4_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_5_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_6_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_7_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_8_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_9_MMStack_Default.ome',
'230816_Hela_JQ1_25uM_9h_10_MMStack_Default.ome'
]

counts_9h = []    
for prefix in prefixes_9h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    counts_9h.append(count_mat)
    del pipe

path = config['analpath']+'230816_Hela_JQ1_25uM_9h_counts.csv'
counts_9h = pd.concat(counts_9h)
counts_9h.to_csv(path)

##################
# Quick t-test
##################

print('2h: ' + str(ttest_ind(counts_0h['counts'],counts_2h['counts'])[1]))
print('4h: ' + str(ttest_ind(counts_0h['counts'],counts_4h['counts'])[1]))
print('6h: ' + str(ttest_ind(counts_0h['counts'],counts_6h['counts'])[1]))
print('9h: ' + str(ttest_ind(counts_0h['counts'],counts_9h['counts'])[1]))

#fig,ax=plt.subplots()
#ax.boxplot([counts_0h['counts'],counts_2h['counts'],counts_4h['counts'],counts_6h['counts'],counts_9h['counts']])

prefixes_2h = [
'230816_Hela_JQ1_40uM_2h_1_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_2_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_3_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_4_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_5_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_6_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_7_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_8_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_9_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_2h_10_MMStack_Default.ome'
]

counts_2h = []
for prefix in prefixes_2h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    counts_2h.append(count_mat)
    del pipe

path = config['analpath']+'230816_Hela_JQ1_40uM_2h_counts.csv'
counts_2h = pd.concat(counts_2h)
counts_2h.to_csv(path)


prefixes_6h = [
'230816_Hela_JQ1_40uM_6h_1_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_2_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_3_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_4_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_5_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_6_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_7_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_8_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_9_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_6h_10_MMStack_Default.ome'
]

counts_6h = []
for prefix in prefixes_6h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    counts_6h.append(count_mat)
    del pipe

path = config['analpath']+'230816_Hela_JQ1_40uM_6h_counts.csv'
counts_6h = pd.concat(counts_6h)
counts_6h.to_csv(path)


prefixes_9h = [
'230816_Hela_JQ1_40uM_9h_1_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_2_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_3_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_4_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_5_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_6_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_7_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_8_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_9_MMStack_Default.ome',
'230816_Hela_JQ1_40uM_9h_10_MMStack_Default.ome'
]

counts_9h = []    
for prefix in prefixes_9h:
    #print("Processing " + prefix)
    pipe = PipelineCount2D(config,prefix)
    pipe.localize(plot=True)
    count_mat = pipe.count()
    counts_9h.append(count_mat)
    del pipe

path = config['analpath']+'230816_Hela_JQ1_40uM_9h_counts.csv'
counts_9h = pd.concat(counts_9h)
counts_9h.to_csv(path)

#fig,ax=plt.subplots()
#ax.boxplot([counts_0h['counts'],counts_2h['counts'],counts_6h['counts'],counts_9h['counts']])
#plt.show()

##################
# Quick t-test
##################

print('2h: ' + str(ttest_ind(counts_0h['counts'],counts_2h['counts'])[1]))
print('6h: ' + str(ttest_ind(counts_0h['counts'],counts_6h['counts'])[1]))
print('9h: ' + str(ttest_ind(counts_0h['counts'],counts_9h['counts'])[1]))

