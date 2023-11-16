import matplotlib.pyplot as plt
from SMLM.figures import Figure_7
import json

prefixes = [
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-G-4-regi_spots-tracked'
]

with open('plot_figure_7.json', 'r') as f:
    config = json.load(f)
    
for prefix in prefixes:
    figure = Figure_7(config,prefix)
    figure.plot()

