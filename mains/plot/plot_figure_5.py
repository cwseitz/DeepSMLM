import matplotlib.pyplot as plt
from SMLM.figures import Figure_5
import json

prefixes = [
'230722_QD-Cal-HighNA-Astigmatism',
'230722_QD-Cal-LowNA-Astigmatism',
'230722_QD-Cal-HighNA-Isotropic',
'230722_QD-Cal-LowNA-Isotropic'
]

with open('plot_figure_5.json', 'r') as f:
    config = json.load(f)

figure = Figure_5(config)
figure.plot(prefixes)

