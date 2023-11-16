import numpy as np
import json
import matplotlib.pyplot as plt
from SMLM.generators import SPAD2D
from skimage.io import imsave
from SMLM.psf.psf2d import crlb2d

with open('run_spad_2d.json', 'r') as f:
    config = json.load(f)

spad2d = SPAD2D(config)
photons, probsum = spad2d.generate(plot=True,r=0)
imsave('/home/cwseitz/Desktop/SPAD.tif',photons)

fig,ax=plt.subplots()
ax.imshow(probsum)
plt.show()

fig,ax=plt.subplots()
ax.imshow(np.sum(photons,axis=0))
plt.show()



