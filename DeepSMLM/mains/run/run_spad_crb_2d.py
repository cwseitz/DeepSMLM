import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.io import imsave
from SMLM.psf.psf2d import crlb2d

"""
Cramer-Rao bound for shot-noise limited regime and SPAD simulation
"""


class CRB2D:
    def __init__(self,cmos_params):
        self.cmos_params = cmos_params
             
    def forward(self,N0space):
        crlb_n0 = np.zeros((len(N0space),4))
        for i,n0 in enumerate(N0space):
            theta0 = np.array([3.5,3.5,0.92,n0])
            crlb_n0[i] = crlb2d(theta0,self.cmos_params)
        return crlb_n0


with open('run_spad_2d.json', 'r') as f:
    config = json.load(f)
    
fig,ax=plt.subplots()
N0space = np.linspace(5,1000,500)
    
gain = 1.0*np.ones((7,7))
offset = np.zeros((7,7))
var = np.zeros((7,7))    
cmos_params = [1.0,1.0,gain,offset,var] 
crb = CRB2D(cmos_params)
crlb_n0_1 = crb.forward(N0space)

gain = 1.0*np.ones((7,7))
offset = 100.0*np.ones((7,7))
var = 5.0*np.ones((7,7))    
cmos_params = [1.0,1.0,gain,offset,var] 
crb = CRB2D(cmos_params)
crlb_n0_2 = crb.forward(N0space)


ax.semilogx(N0space,crlb_n0_1[:,0]*108.3,color='gray',label='SPAD')
ax.semilogx(N0space,crlb_n0_2[:,0]*108.3,color='black',label='CMOS')

ax.set_xlabel('Intensity (photons)')
ax.set_ylabel('Localization uncertainty (nm)')
ax.legend()
ax.set_title('Theoretical CRLB')
plt.show()
    



