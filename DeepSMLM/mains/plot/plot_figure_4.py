import matplotlib.pyplot as plt
from SMLM.figures import Figure_4
import json

prefix = '230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4_spots'

with open('plot_figure_4.json', 'r') as f:
    config = json.load(f)
    
figure = Figure_4(config,prefix)
figure.plot()
plt.show()
