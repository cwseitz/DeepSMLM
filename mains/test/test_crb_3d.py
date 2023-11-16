from SMLM.tests import *
import matplotlib.pyplot as plt
import json

with open('test_crb_3d.json', 'r') as f:
    config = json.load(f)

crb3dtest = CRB3D_Test2(config,max_iters=100,error_samples=1000)
fig,(ax1,ax2) = plt.subplots(1,2)
crb3dtest.plot(ax1,ax2)
plt.show()
