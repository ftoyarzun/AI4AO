# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 19:20:54 2025

@author: franc
"""

import torch

from PhaseDataset import PhaseDataset
from SimuEnd2EndWFS import End2EndWFS, AOLoop, CheckpointManager
from mmengine import Config
import numpy as np
import matplotlib.pyplot as plt
import os




if __name__ == "__main__":
    
    device = 'cuda' # set to "cpu" if Cuda is not available
    
    paramfile = 'params_exp.py'  # file of experimental parameters

    gif_path = 'closed_loop_test_mod10.gif'
    # writer = imageio.get_writer(gif_path, mode="i", fps = 10)

    # Config extraction
    AtmosParams = Config.fromfile(paramfile)['AtmosParams']
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']
    TrainParams = Config.fromfile(paramfile)['TrainParams']

    # Dataset creation
    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
    

    
    # Initialisation of the system 
    gain = 0.2
    start_after_iteration = 0
    
    z_FullRes = dataset.z_FullRes.to(device, dtype=torch.float32).view(-1, dataset.Nzernike).transpose(0, 1)
    #dataset.LoadTestMovingWavefront()
    phaseGT,zernike,photons,ron,r0s,wind,fractionalr0 = dataset.GetMovingWavefront()
    
    
    
    Trained_End2EndWFS = End2EndWFS (WFSParams,device, "SimpleNet")
    Trained_End2EndWFS.eval()
    checkpoint_path = 'PyramidOHP209.pth'
    checkpointManager = CheckpointManager(Trained_End2EndWFS, WFSParams, TrainParams, checkpoint_path)
    checkpointManager.load_network(should_load_optimizer = False)
    #checkpointManager.load_free_mask(should_load_optimizer = False)
    
    Trained_AO_Loop = AOLoop(Trained_End2EndWFS, z_FullRes, gain, phaseGT, zernike, photons, ron, start_after_iteration = start_after_iteration)
    
    Pyramid_End2EndWFS_mod0 = End2EndWFS (WFSParams,device, "Linear")
    Pyramid_End2EndWFS_mod0.WFSmodule.WFS.maskType = "Pyramid"
    Pyramid_AO_Loop_mod0 = AOLoop(Pyramid_End2EndWFS_mod0, z_FullRes.to(device), gain, phaseGT, zernike, photons, ron, start_after_iteration = start_after_iteration, modulation = 1)
    
    Pyramid_End2EndWFS_mod5 = End2EndWFS (WFSParams,device, "Linear")
    Pyramid_End2EndWFS_mod5.WFSmodule.WFS.maskType = "Pyramid"
    Pyramid_AO_Loop_mod5 = AOLoop(Pyramid_End2EndWFS_mod5, z_FullRes.to(device), gain, phaseGT, zernike, photons, ron, start_after_iteration = start_after_iteration, modulation = 5)
        
    
    
    
#%%    
    
    
    
    
    with torch.no_grad():
    
        
        
        sample_to_look = 0
        
        fig, ax = plt.subplots(1, 6, figsize=(20, 5))
        img1 = ax[0].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img2 = ax[1].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img3 = ax[2].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img4 = ax[3].imshow(Trained_AO_Loop.images[sample_to_look,:,:].cpu().detach().numpy())
        plot1, = ax[4].plot(zernike[sample_to_look,:].cpu().detach().numpy())
        plot2, = ax[4].plot(zernike[sample_to_look,:].cpu().detach().numpy())
        plot3, = ax[4].plot(zernike[sample_to_look,:].cpu().detach().numpy())
        plot21, = ax[4].plot(zernike[sample_to_look,:].cpu().detach().numpy())
        plot4, = ax[5].semilogy(zernike[sample_to_look,:].cpu().detach().numpy())
        plot5, = ax[5].semilogy(zernike[sample_to_look,:].cpu().detach().numpy())
        plot6, = ax[5].semilogy(zernike[sample_to_look,:].cpu().detach().numpy())
        plt.subplots_adjust(left=0.02, right=0.98, wspace=0.1)
        # plt.subplots_adjust(left = 0.1, top = 0.9, right = 0.9, bottom = 0.1, hspace = 0.5, wspace = 0.5)

        
        
        print(f'Photons = {photons[sample_to_look,:,:].item()}, RON = {ron[sample_to_look,:,:].item()}')
        
        for i in range(500):
            # Get new WFS images after applying the correction
            phaseGT,zernike,_,_,_,_,_ = dataset.GetMovingWavefront()
            
            Trained_AO_Loop.step(phaseGT)
            Pyramid_AO_Loop_mod0.step(phaseGT)
            Pyramid_AO_Loop_mod5.step(phaseGT)
            
            if i % 10 == 0:
                plt.subplot(1,6,1)
                plt.title(f'Input phase r0 = {r0s[sample_to_look,:].item():.3f}')
                img1.set_data(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
                img1.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
                
                
                plt.subplot(1,6,2)
                plt.title('Reconstructed phase')
                img2.set_data(Trained_AO_Loop.z_reconstructed[sample_to_look,:,:].cpu().detach().numpy())
                img2.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
                
                plt.subplot(1,6,3)
                plt.title('Residual phase')
                img3.set_data(Trained_AO_Loop.residual_phase[sample_to_look,:,:].cpu().detach().numpy())
                img3.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
                
                plt.subplot(1,6,4)
                plt.title(f'Photons = {photons[sample_to_look,:,:].item():.0f}, RON = {ron[sample_to_look,:,:].item():.2f}')
                img4.set_data(Trained_AO_Loop.images[sample_to_look,:,:].cpu().detach().numpy())
                img4.set_clim(vmin=np.min(img4.get_array()), vmax=np.max(img4.get_array()))
                
                
                plt.subplot(1, 6, 5)
                plt.title(f"Iteration number {i}")
                plot1.set_ydata(zernike[sample_to_look,:].cpu().detach().numpy())
                plot2.set_ydata(Trained_AO_Loop.z_estimated[sample_to_look,:].cpu().detach().numpy())
                plot21.set_ydata(Pyramid_AO_Loop_mod0.z_estimated[sample_to_look,:].cpu().detach().numpy())
                plot3.set_ydata(Pyramid_AO_Loop_mod5.z_estimated[sample_to_look,:].cpu().detach().numpy())
                plt.legend(['Ground truth', 'Reconstruction', 'Difference'])
                plt.xlabel('Zernike mode')
                plt.ylabel('Mode amplitude')
                plt.ylim([-1.5, 1.5])
                plt.pause(0.5)
                
                
                plt.subplot(1, 6, 6)
                plt.title("Residual variance")
                plot4.set_data(np.arange(Trained_AO_Loop.residual_variance.shape[1]), Trained_AO_Loop.residual_variance[sample_to_look,:].cpu().detach().numpy())
                plot5.set_data(np.arange(Pyramid_AO_Loop_mod5.residual_variance.shape[1]), Pyramid_AO_Loop_mod5.residual_variance[sample_to_look,:].cpu().detach().numpy())
                plot6.set_data(np.arange(Pyramid_AO_Loop_mod0.residual_variance.shape[1]), Pyramid_AO_Loop_mod0.residual_variance[sample_to_look,:].cpu().detach().numpy())
                ax[5].relim()
                ax[5].autoscale_view()
                plt.xlabel('Iteration')
                plt.ylabel('Residual variance', labelpad=10)
                plt.legend(['NN', 'Pyramid mod 5', 'Pyramid mod 0'])
                plt.pause(0.1)
                
                fig.canvas.draw()
            
            # if i % 2 == 0:
            #     image = np.array(fig.canvas.buffer_rgba())
            #     writer.append_data(image)
            
            
            if i % 50 == 0:
                sample_to_look += 1
            
        # writer.close()
        
    tested_r0s = np.array([0.25, 0.15, 0.05])
    tested_photons = np.array([6, 5, 4, 3.5])
    NN_res_var = np.array([[0.0400, 0.0394, 0.0447, 0.0899],
                  [0.0864, 0.0838, 0.0940, 0.1386],
                  [0.5408, 0.5487, 0.5527, 0.6681]])
    pyr0_res_var = np.array([[0.0373, 0.0394, 0.0765, 0.3801],
                  [0.0921, 0.0901, 0.1321, 0.1811],
                  [5.4841, 5.6230, 5.4893, 6.3268]])
    pyr5_res_var = np.array([[0.0450, 0.0393, 0.0523, 0.0525],
                  [0.1440, 0.1503, 0.1490, 0.4346],
                  [0.5806, 0.5802, 0.6034, 0.9458]])