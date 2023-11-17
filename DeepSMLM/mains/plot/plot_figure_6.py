import matplotlib.pyplot as plt
from SMLM.figures import Figure_3
import json

prefixes = [
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4'
]

with open('plot_figure_6.json', 'r') as f:
    config = json.load(f)
    
for prefix in prefixes:
    figure = Figure_6(config)
    figure.plot(prefix)
    plt.show()
