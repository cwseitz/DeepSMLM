import matplotlib.pyplot as plt
from SMLM.figures import Figure_2
import json

with open('plot_figure_2.json', 'r') as f:
    config = json.load(f)
    
prefix = '230720_Mix3D_k1'
figure = Figure_2(config)
figure.plot(prefix)
plt.show()
