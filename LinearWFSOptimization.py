# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:57:52 2025

@author: franc
"""

import torch
import torch.nn as nn
from mmengine import Config
from PhaseDataset import PhaseDataset
from TorchPropagator import WFS
import pylab as plt
import numpy as np

from tqdm import tqdm

import os

from MaskGeneration import MaskManager

import torch.nn.functional as F




class Sensitivity_Loss(nn.Module):
    def __init__(self, pupil):
        super().__init__()
        
        self.Nsap_sqrt = pupil.sum().sqrt()
        self.Nsap = pupil.sum()
        self.RON = 10
        self.Nphotons = 1e4
        
    # def forward(self, x):
    #     sens = torch.std(x, dim = (-2, -1))
    #     return -torch.median(sens)
    
    def forward(self, x, I0):

        x_RON, x_PN, non_diag = self.ComputeSensitivities(x, I0)
        
        return -torch.mean(x_RON) - torch.mean(x_PN)# + torch.std(non_diag)*1e1

    
    def ComputeSensitivities(self, x, I0):
        x = x.flatten(1)
        
        
        x_RON = x @ x.T
        
        non_diag_terms = x_RON - torch.diag(torch.diag(x_RON))
        
        x_RON = torch.diag(x_RON)
        x_RON = torch.sqrt(x_RON) * self.Nsap_sqrt
        
        
        I0 = I0.flatten().unsqueeze(0)
        
        x_PN = (x / I0) @ x.T
        x_PN = torch.diag(x_PN)
        x_PN = torch.sqrt(x_PN)
        
        return x_RON, x_PN, non_diag_terms


class DynamicRange_Loss(nn.Module):
    def __init__(self, nModes):
        super().__init__()
        
        self.target = torch.eye(nModes, device = device)
        self.nModes = nModes
    
    def forward(self, iMat):

        reconstruction = self.ComputeDynamics(iMat)
        return torch.mean(   (self.target - reconstruction) ** 2  )# - torch.mean(x_PN) + torch.std(non_diag)*1e7
    
    def ComputeDynamics(self, iMat):
        
        R = torch.linalg.pinv(iMat.view(self.nModes,-1)).T
        
        epsilon = 1e0
        
        push = wfs.Propagator(z_FullRes * epsilon)
        pull = wfs.Propagator(-z_FullRes * epsilon)
        
        iMat2 = (push - pull) / 2 / epsilon
        
        reconstruction = R @ iMat2.view(self.nModes,-1).T
        
        return reconstruction
    
    
class Linear_Loss(nn.Module):
    def __init__(self, nModes):
        super().__init__()
        
        self.target = torch.eye(nModes, device = device)
        self.nModes = nModes
    
    def forward(self, iMat):

        reconstruction = self.ComputeDynamics(iMat)
        return torch.mean(   (self.target - reconstruction) ** 2  )# - torch.mean(x_PN) + torch.std(non_diag)*1e7
    
    def ComputeDynamics(self, iMat):
        
        R = torch.linalg.pinv(iMat.view(self.nModes,-1)).T
        
        epsilon = 1e0
        
        push = wfs.Propagator(z_FullRes * epsilon)
        pull = wfs.Propagator(-z_FullRes * epsilon)
        
        iMat2 = (push - pull) / 2 / epsilon
        
        reconstruction = R @ iMat2.view(self.nModes,-1).T
        
        return reconstruction


def remove_tip_tilt(mask):
    # Least-squares fit to remove linear plane (tip/tilt)
    coeffs = torch.linalg.lstsq(uv_coords, mask).solution
    tilt_plane = uv_coords @ coeffs
    return mask - tilt_plane


def TrainLinearWFS():
    
    # fig,ax = plt.subplots(figsize = (10,10))
    # # maskGenerator.update_masks()
    # # img = ax.imshow(maskGenerator(uv_coords).view(N,N).cpu().detach().numpy())
    # # img = ax.imshow(maskGenerator.transmisionMask[0,0].cpu().detach())
    # img = ax.imshow(mask.cpu().detach())
    # fig.colorbar(img)
    # plt.show()
    
    # maskGenerator.train()
    
    progressBar = tqdm(range(300))
    
    
    
    
    for ii in progressBar:
        total_loss = 0
        optimizer.zero_grad()
    
        
        # mask = remove_tip_tilt(maskGenerator(uv_coords))
        # mask = mask.view(N, N)  
        
        wfs.SetMask(mask)
        
        # maskGenerator.update_masks()
        
        I0 = wfs.Propagator(z_FullRes[0] * 0)
        
        epsilon = 1e-5
        
        push = wfs.Propagator(z_FullRes * epsilon)
        pull = wfs.Propagator(-z_FullRes * epsilon)
        
        iMat = (push - pull) / 2 / epsilon
        
        l = loss(iMat, I0)
        l1 = torch.log(loss_dynamic(iMat))
        
        total_loss += l
        total_loss += l1
    
        total_loss.backward()
        
        optimizer.step()
        
        if ii % 20 == 0:
            #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {ii}, train loss : {total_loss.item():.5f}")
            # print(f" Run n°  {ii}, train loss : {total_loss.item():.5f}, params : {maskGenerator.param.tolist()[0]:.5f}")
            # img.set_data(maskGenerator.transmisionMask[0,1].cpu().detach())
            #img.set_data(mask.cpu().detach())
            #img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            
            #fig.canvas.draw()
            # image = np.array(fig.canvas.buffer_rgba())
            # writer.append_data(image)
            
            #plt.pause(0.1)
            pass
    


if __name__ == "__main__":
    
    device = 'cuda' # set to "cpu" if Cuda is not available
    
    paramfile = 'params_exp.py'  # file of experimental parameters

    gif_path = 'test_mask_animation.gif'
    
    mask_path = 'Pyramid_mask.pth'
    
    AtmosParams = Config.fromfile(paramfile)['AtmosParams']
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']
    TrainParams = Config.fromfile(paramfile)['TrainParams']
    
    WFSParams["useNoise"] = False

    # Dataset creation
    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
    z_FullRes = dataset.z_FullRes.permute(2,0,1)
    
    wfs = WFS(WFSParams, device)
    wfs.modulation = 0
    
    maskGenerator = MaskManager(WFSParams, device, wfs).to(device)
        
    
    
    
    
    
    # N = WFSParams["Nres"]*2
    # u = torch.linspace(-N//2, N // 2 - 1, N, device = device) / (N/2)  # Normalized frequency range
    # U, V = torch.meshgrid(u, u, indexing="xy")  # Create the full grid
    # uv_coords = torch.stack([U.flatten(), V.flatten()], dim=1)
    N = int(WFSParams["Nres"]*WFSParams["sampling"])

   
    mask_path = 'mask.pth'

    # if os.path.exists(mask_path):
    #     loaded_mask_tensor = torch.load(mask_path, map_location=device)
    #     mask = nn.Parameter(loaded_mask_tensor, requires_grad=True)
    #     print("Mask loaded successfully.")
    # else:
    #     print(f"Mask file '{mask_path}' does not exist. Initializing a new mask.")
    #     mask = nn.Parameter(torch.randn(size=(N, N), device=device, dtype=torch.float32) * 1e-2, requires_grad=True)
    
    mask = nn.Parameter(torch.randn(size=(N, N), device=device, dtype=torch.float32) * 1e-2, requires_grad=True)
        
    # optimizer = torch.optim.AdamW(maskGenerator.parameters(),0.01, weight_decay=1, fused=True)
    optimizer = torch.optim.AdamW([mask],0.01, fused=True)
    loss = Sensitivity_Loss(wfs.pupil)
    loss_dynamic = DynamicRange_Loss(z_FullRes.shape[0])

    
    # a = time.time()
    train_loss = TrainLinearWFS()
    # b = time.time() - a 
    # a,b,c = loss.ComputeSensitivities(iMat, I0)
    # torch.save(mask.data, 'mask.pth')

