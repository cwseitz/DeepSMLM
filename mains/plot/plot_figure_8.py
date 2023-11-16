import matplotlib.pyplot as plt
from SMLM.figures import Figure_8
import json

prefixes = [
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4_spots_clustered',
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4_spots_clustered_ROI0',
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4_spots_clustered_ROI1',
'230726_Hela_J549_1pM_1hr_j646_10pM_overnight_5mW_20mW-R-4_spots_clustered_ROI2'
]

with open('plot_figure_3.json', 'r') as f:
    config = json.load(f)
    
centers = [(130,145),(150,180),(90,175)]
figure = Figure_8(config)
figure.plot(prefixes,centers)
plt.show()
