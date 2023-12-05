class LocalizationVAE3(nn.Module):
    """Includes a pre-trained UNET encoder"""
    def __init__(self, latent_dim, nx, ny, scaling_factor=800.0):
        super(LocalizationVAE3, self).__init__()
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
             self.train_config = json.load(train_config)
        args = train_config['arch']['args']
        cnn = UNetModel(args['n_channels'],args['n_features'])
        cnn = model.to(device=device)
        checkpoint = torch.load(modelpath+modelname, map_location=device)
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

        return x,mu,logvar

        


