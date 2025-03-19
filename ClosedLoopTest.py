# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:26:38 2025

@author: franc
"""

import torch

from PhaseDataset import PhaseDataset, PermanentPhaseDataset
from SimuEnd2EndWFS import End2EndWFS
from mmengine import Config
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from TorchPropagator import Zernike
import copy

import imageio.v2 as imageio

import torch.nn as nn



if __name__ == "__main__":
    
    device = 'cuda' # set to "cpu" if Cuda is not available
    
    paramfile = 'params_exp.py'  # file of experimental parameters

    gif_path = 'closed_loop_test.gif'
    writer = imageio.get_writer(gif_path, mode="i", fps = 10)

    # Config extraction
    AtmosParams = Config.fromfile(paramfile)['AtmosParams']
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']
    TrainParams = Config.fromfile(paramfile)['TrainParams']

    # Dataset creation
    dataset = PhaseDataset(WFSParams['D'],WFSParams['Nres'],WFSParams['Nzernike'],                       
                           AtmosParams['L0'],AtmosParams['r0'],AtmosParams['Nphases'],
                           WFSParams['Nactuator'],LoopParams['levelOfCorrection'],
                           LoopParams['loopFrequency'], LoopParams['delayFrames'], LoopParams['windSpeedVector'],
                           WFSParams['Nphotons'], WFSParams['RON'], device)
    
    
    permanentPhaseDataset = PermanentPhaseDataset()
    
    # Initialisation of the system 
    
    Trained_End2EndWFS = End2EndWFS (WFSParams,device)
    
    
    checkpoint_path = 'FreeTest4.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        Trained_End2EndWFS.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming from epoch", start_epoch)
    else:
        start_epoch = 0
        print("Starting from scratch")
        
    
    z_FullRes = dataset.z_FullRes.to(device, dtype=torch.float32).view(-1, 50).transpose(0, 1)
    
    Trained_End2EndWFS.eval()
    
    nLayers = 10
    loopPeriod = 1/LoopParams['loopFrequency']
    
    with torch.no_grad():
    
        dataset.ResetMovingWavefront()
        phaseGT,zernike,photons,ron,r0s,wind,fractionalr0 = dataset.GetMovingWavefront(nLayers, loopPeriod)
        Trained_End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons, ron)

        # **Closed-loop iterative correction**
        z_estimated = torch.zeros_like(zernike)  # Start with zero correction
        
        z_reconstructed = torch.zeros_like(phaseGT)
        residual_phase = torch.zeros_like(phaseGT)  # Start with the original phase
        
        Trained_End2EndWFS(phaseGT)
        images = Trained_End2EndWFS.WFSmodule(phaseGT)
        
        gain = 0.3
        sample_to_look = 0
        
        fig, ax = plt.subplots(1,5, figsize = (24,5))
        img1 = ax[0].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img2 = ax[1].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img3 = ax[2].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img4 = ax[3].imshow(images[sample_to_look,:,:].cpu().detach().numpy())
        plot1, = ax[4].plot(zernike[sample_to_look,:].cpu().detach().numpy())
        plot2, = ax[4].plot(zernike[sample_to_look,:].cpu().detach().numpy())
        plot3, = ax[4].plot(zernike[sample_to_look,:].cpu().detach().numpy())
        
        
        print(f'Photons = {photons[sample_to_look,:,:].item()}, RON = {ron[sample_to_look,:,:].item()}')
        
        for i in range(500):
            
            # Get new WFS images after applying the correction
            phaseGT,zernike,_,_,_,_,_ = dataset.GetMovingWavefront(nLayers, loopPeriod)
            residual_phase = phaseGT - z_reconstructed
            # Predict Zernike coefficients and update estimate
            z_output = Trained_End2EndWFS(residual_phase)
            z_estimated = z_estimated + gain * z_output  # Apply correction with gain

            # Convert Zernike coefficients to full-resolution wavefront
            z_reconstructed = torch.matmul(z_estimated, z_FullRes).view_as(phaseGT)
            
            images = Trained_End2EndWFS.WFSmodule(residual_phase)
            
            plt.subplot(1,5,1)
            plt.title(f'Input phase r0 = {r0s[sample_to_look,:].item():.3f}')
            img1.set_data(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
            img1.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
            
            
            plt.subplot(1,5,2)
            plt.title('Reconstructed phase')
            img2.set_data(z_reconstructed[sample_to_look,:,:].cpu().detach().numpy())
            img2.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
            
            plt.subplot(1,5,3)
            plt.title('Residual phase')
            img3.set_data(residual_phase[sample_to_look,:,:].cpu().detach().numpy())
            img3.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
            
            plt.subplot(1,5,4)
            plt.title(f'Photons = {photons[sample_to_look,:,:].item():.0f}, RON = {ron[sample_to_look,:,:].item():.2f}')
            img4.set_data(images[sample_to_look,:,:].cpu().detach().numpy())
            img4.set_clim(vmin=np.min(img4.get_array()), vmax=np.max(img4.get_array()))
            
            
            plt.subplot(1, 5, 5)
            plt.title(f"Iteration number {i}")
            plot1.set_ydata(zernike[sample_to_look,:].cpu().detach().numpy())
            plot2.set_ydata(z_estimated[sample_to_look,:].cpu().detach().numpy())
            plot3.set_ydata((zernike - z_estimated)[sample_to_look,:].cpu().detach().numpy())
            plt.legend(['Ground truth', 'Reconstruction', 'Difference'])
            plt.xlabel('Zernike mode')
            plt.ylabel('Mode amplitude')
            plt.ylim([-1.5, 1.5])
            plt.pause(0.1)
            
            fig.canvas.draw()
            
            if i % 2 == 0:
                image = np.array(fig.canvas.buffer_rgba())
                writer.append_data(image)
            
            
            if i % 50 == 0:
                sample_to_look += 1
            
        writer.close()
        
        
        
        
        
    
    
    