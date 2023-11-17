import matplotlib.pyplot as plt
from SMLM.figures import Figure_9
import json

prefix = '230909_Hela_j646_10pM_overnight_2000frames_20mW_live-3'


with open('plot_figure_9.json', 'r') as f:
    config = json.load(f)
    
figure = Figure_9(config)
figure.plot1(prefix)
figure.plot2()
plt.show()
