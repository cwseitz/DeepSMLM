import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
from pathlib import Path
from SMLM.localize import LoGDetector
from SMLM.psf.psf3d import MLE3D, MLE3D_MCMC, hessiso_auto3d
from numpy.linalg import inv

class PipelineMLE3D_MCMC:
    def __init__(self,config,dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
        self.dump_config()
    def dump_config(self):
        with open(self.analpath+self.dataset.name+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)        
    def localize(self,plot=False,tmax=None,iters=5):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        if tmax is not None: nt = tmax
        threshold = self.config['thresh_log']
        spotst = []
        if not file.exists():
            for n in range(nt):
                print(f'Det in frame {n}')
                framed = self.stack[n]
                log = LoGDetector(framed,threshold=threshold)
                spots = log.detect() #image coordinates
                if plot:
                    log.show(); plt.show()
                spots = self.fit(framed,spots)
                spots = spots.assign(frame=n)
                spotst.append(spots)
            spotst = pd.concat(spotst)
            self.save(spotst)
        else:
            print('Spot files exist. Skipping')
        return spotst
        
    def get_errors(self,theta,adu):
        hess = hessiso_auto3d(theta,adu,self.cmos_params)
        try:
            errors = np.sqrt(np.diag(inv(hess)))
        except:
            errors = np.empty((4,))
            errors[:] = np.nan
        return errors
        
    def fit(self,frame,spots,plot=False,patchw=3):
        lr = np.array([0.01,0.01,0.01,350.0])
        spots['x_mle'] = None; spots['y_mle'] = None; spots['N0'] = None;
        spots['x_err'] = None; spots['y_err'] = None; spots['s_err'] = None; spots['N0_err'] = None;
        for i in spots.index:
            print(f'Fitting spot {i}')
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.cmos_params[3]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,self.config['sigma'],self.config['N0']])
            opt = MLE3D_MCMC(theta0,adu,self.config) #cartesian coordinates with top-left origin
            theta_mle, loglike, post_samples = opt.optimize(iters=100,plot=False,lr=lr)
            dx = theta_mle[1] - patchw; dy = theta_mle[0] - patchw
            spots.at[i, 'x_mle'] = x0 + dx
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'z_mle'] = theta_mle[2]
            spots.at[i, 'N0'] = theta_mle[3]
            
            spots.at[i, 'x_mcmc_avg'] = np.mean(post_samples[1,:])
            spots.at[i, 'y_mcmc_avg'] = np.mean(post_samples[0,:])
            spots.at[i, 'z_mcmc_avg'] = np.mean(post_samples[2,:])
            spots.at[i, 'N0_mcmc_avg'] = np.mean(post_samples[3,:])
            spots.at[i, 'x_mcmc_std'] = np.std(post_samples[1,:])
            spots.at[i, 'y_mcmc_std'] = np.std(post_samples[0,:])
            spots.at[i, 'z_mcmc_std'] = np.std(post_samples[2,:])
            spots.at[i, 'N0_mcmc_std'] = np.std(post_samples[3,:])
 
        return spots
    def save(self,spotst):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        spotst.to_csv(path)

class PipelineMLE3D:
    def __init__(self,config,dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
        self.dump_config()
    def dump_config(self):
        with open(self.analpath+self.dataset.name+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)        
    def localize(self,plot=False,tmax=None,iters=5,patchw=3,lr=None):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        if tmax is not None: nt = tmax
        threshold = self.config['thresh_log']
        spotst = []
        if not file.exists():
            for n in range(nt):
                print(f'Det in frame {n}')
                framed = self.stack[n]
                print(self.dataset.theta[n])
                log = LoGDetector(framed,threshold=threshold)
                spots = log.detect() #image coordinates
                if plot:
                    log.show(); plt.show()
                spots = self.fit(framed,spots,lr=lr,patchw=patchw)
                spots = spots.assign(frame=n)
                spotst.append(spots)
            spotst = pd.concat(spotst)
            self.save(spotst)
        else:
            print('Spot files exist. Skipping')
        return spotst
        
    def get_errors(self,theta,adu):
        hess = hessiso_auto3d(theta,adu,self.cmos_params)
        try:
            errors = np.sqrt(np.diag(inv(hess)))
        except:
            errors = np.empty((4,))
            errors[:] = np.nan
        return errors
        
    def fit(self,frame,spots,plot=False,patchw=3,lr=None):
        if not lr:
            lr = np.array([0.01,0.01,0.01,350.0])
        spots['x_mle'] = None; spots['y_mle'] = None; spots['N0'] = None;
        spots['x_err'] = None; spots['y_err'] = None; spots['s_err'] = None; spots['N0_err'] = None;
        for i in spots.index:
            print(f'Fitting spot {i} with patchw = {patchw}')
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.cmos_params[3]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,0.0,self.config['N0']])
            opt = MLE3D(theta0,adu,self.config) #cartesian coordinates with top-left origin
            theta_mle, loglike = opt.optimize(max_iters=1000,plot=False,lr=lr)
            dx = theta_mle[1] - patchw; dy = theta_mle[0] - patchw
            spots.at[i, 'x_mle'] = x0 + dx
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'z_mle'] = theta_mle[2]
            spots.at[i, 'N0'] = theta_mle[3]
            print(f'Fit params {x0+dx,y0+dy,theta_mle[2],theta_mle[3]}')
        return spots
    def save(self,spotst):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        spotst.to_csv(path)


