import matplotlib.pyplot as plt
from SMLM.figures import Figure_0
import json

with open('plot_figure_0.json', 'r') as f:
    config = json.load(f)

prefix = '220926_Dark-Noise-50ms-100frames-1'

figure = Figure_0(config,prefix)
figure.plot()
plt.show()
