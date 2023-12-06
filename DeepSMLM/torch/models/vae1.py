import torch, math, copy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import json
from torchvision import datasets, transforms
from scipy.stats import kde
from .localize import LocalizationCNN
from BaseSMLM.generators import Generator
        
class Decoder(nn.Module):
    def __init__(self, nx, ny):
        super(Decoder, self).__init__()
        self.nx = nx
        self.ny = ny

    def lamx(self, X, x0, sigma):
        alpha = torch.sqrt(torch.tensor(2.0)) * sigma
        return 0.5 * (torch.erf((X + 0.5 - x0) / alpha) - torch.erf((X - 0.5 - x0) / alpha))

    def lamy(self, Y, y0, sigma):
        alpha = torch.sqrt(torch.tensor(2.0)) * sigma
        return 0.5 * (torch.erf((Y + 0.5 - y0) / alpha) - torch.erf((Y - 0.5 - y0) / alpha))

    def forward(self, z, sigma=0.92, texp=1.0, eta=1.0, N0=1000.0, patch_hw=3):
        batch_size, _ = z.shape
        nspots = _ // 2
        z = z.reshape((batch_size,2,nspots))
        x = torch.arange(0, 2 * patch_hw, dtype=torch.float32).cuda()
        y = torch.arange(0, 2 * patch_hw, dtype=torch.float32).cuda()
        X, Y = torch.meshgrid(x, y)
        mu = torch.zeros((batch_size, 1, self.nx, self.ny), dtype=torch.float32).cuda()

        i0 = eta * N0 * texp

        for batch_idx in range(batch_size):
            for n in range(nspots):
                x0, y0 = z[batch_idx,:,n]
                x0r = torch.round(x0).int()
                y0r = torch.round(y0).int()
                patchx, patchy = x0r - patch_hw, y0r - patch_hw
                x0p = x0 - patchx
                y0p = y0 - patchy
                if 0 <= patchx < self.nx - 2 * patch_hw and 0 <= patchy < self.ny - 2 * patch_hw:
                    this_mu = i0 * self.lamx(X, x0p, sigma) * self.lamy(Y, y0p, sigma)
                    mu[batch_idx, :, patchx:patchx + 2 * patch_hw, patchy:patchy + 2 * patch_hw] += this_mu


        return mu

class LocalizationVAE1(nn.Module):
    """Includes a pre-trained DeepSTORM encoder"""
    def __init__(self, latent_dim, nx, ny, scaling_factor=800.0):
        super(LocalizationVAE1, self).__init__()
        self.nx = nx; self.ny = ny
        self.latent_dim = latent_dim
        nc=1; ndf=8; nz=1

        self.flatten = nn.Flatten()

        # Fully connected layers for mu and logvar
        self.fcdim = (4*self.nx)**2
        hidden1 = 256; hidden2 = 128
        
        self.mu = nn.Sequential(
            nn.Linear(self.fcdim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, latent_dim)
        )

        self.logvar = nn.Sequential(
            nn.Linear(self.fcdim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, latent_dim)
        )
                
        self.decoder = Decoder(nx,ny)
        self.norm = nn.BatchNorm1d(self.fcdim)
        
    def load_cnn(self,modelpath,modelname):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_config_path = modelpath+modelname+'/'+modelname+'.json'
        with open(train_config_path,'r') as train_config:
             train_config = json.load(train_config)
        args = train_config['arch']['args']
        cnn = LocalizationCNN(args['nz'],args['scaling_factor'],dilation_flag=args['dilation_flag'])
        cnn = cnn.to(device=device)
        checkpoint = torch.load(modelpath+modelname+'/'+modelname+'.pth', map_location=device)
        cnn.load_state_dict(checkpoint['state_dict'])
        cnn.eval()
        self.encoder = cnn

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + (eps * std) + self.nx/2
        return z

    def encode(self, input):
        conv = self.encoder(input)
        blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
        conv = blur(conv)
        return conv

    def decode(self, z):
        out = self.decoder.forward(z)
        return out

    def forward(self, input):
        conv = self.encode(input)
        out = self.flatten(conv)
        mu = self.mu(out)
        logvar = self.logvar(out)
        z = self.sample(mu,logvar)
        x = self.decode(z)

        #zplot = z[0].reshape((2,5)).cpu().detach().numpy()
        #fig,ax=plt.subplots(1,3)
        #ax[0].imshow(input[0,0].cpu().detach().numpy())
        #ax[1].imshow(x[0,0].cpu().detach().numpy())
        #ax[1].scatter(zplot[1,:],zplot[0,:])
        #ax[2].imshow(conv[0,0].cpu().detach().numpy())
        #plt.show()

        return x,conv,mu,logvar

        


