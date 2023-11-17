import matplotlib.pyplot as plt
from SMLM.figures import Figure_1
import json

with open('plot_figure_1.json', 'r') as f:
    config = json.load(f)
    
figure = Figure_1(config)


prefixes2d_k = [
'230720_Mix2D_k1_adu',
'230720_Mix2D_k3_adu',
'230720_Mix2D_k5_adu'
]

prefixes2d_p = [
'230720_Mix2D_p100_adu',
'230720_Mix2D_p500_adu',
'230720_Mix2D_p1000_adu'
]

figure.plot2d(prefixes2d_p,prefixes2d_k)
plt.show()

prefixes3d_k = [
'230720_Mix3D_k1_adu',
'230720_Mix3D_k3_adu',
'230720_Mix3D_k5_adu'
]

prefixes3d_p = [
'230720_Mix3D_p1000_adu',
'230720_Mix3D_p10000_adu',
'230720_Mix3D_p100000_adu'
]

figure.plot3d(prefixes3d_p,prefixes3d_k)
plt.show()

