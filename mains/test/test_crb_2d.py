from SMLM.tests import CRB2D_Test1
import matplotlib.pyplot as plt
import json

with open('iso2d.json', 'r') as f:
    setup_params = json.load(f)
crb2dtest = CRB2D_Test1(setup_params)
fig,ax = plt.subplots()
crb2dtest.plot(ax)
plt.show()
