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
    AtmosParams["Nphases"] = 8
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']
    TrainParams = Config.fromfile(paramfile)['TrainParams']

    # Dataset creation
    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
    

    
    # Initialisation of the system 
    gain = 0.3
    leak = 0.99
    start_after_iteration = 0
    
    z_FullRes = dataset.z_FullRes.view(-1, dataset.Nzernike).transpose(0, 1)
    #dataset.LoadTestMovingWavefront()
    phaseGT,zernike,photons,ron,r0s,wind,fractionalr0 = dataset.GetMovingWavefront(generateClosedLoop = True)
    
    #phaseGT[0] = phase_to_estimate
    
    Trained_End2EndWFS = End2EndWFS (WFSParams, AtmosParams, device)
    Trained_End2EndWFS.eval()
    checkpoint_path = 'Papyrus_2nd_Stage.pth'
    checkpointManager = CheckpointManager(Trained_End2EndWFS, WFSParams, TrainParams, checkpoint_path)
    # checkpointManager.load(should_load_optimizer = False)
    # checkpointManager.load_parametric_mask('Papyrus.pth', should_load_optimizer = False)
    checkpointManager.load_network(should_load_optimizer = False)
    # checkpointManager.load_parametric_mask(should_load_optimizer = False)
    # checkpointManager.load_free_transmisionMask(should_load_optimizer = False)
    
    Trained_End2EndWFS.maskManager.update_masks()
    Trained_End2EndWFS.WFS.BuildReferenceIntensity()
    
    Trained_AO_Loop = AOLoop(WFSParams, Trained_End2EndWFS, z_FullRes, gain, leak, phaseGT, zernike, photons, ron, start_after_iteration = start_after_iteration)
    
    WFSParams_Linear = Config.fromfile(paramfile)['WFSParams']
    WFSParams_Linear["Reconstruction"] = "Linear"
    
    End2EndWFS_Linear = End2EndWFS (WFSParams_Linear,AtmosParams,device)
    End2EndWFS_Linear.eval()
    AO_Loop_Linear = AOLoop(WFSParams_Linear, End2EndWFS_Linear, z_FullRes, gain, leak,  phaseGT, zernike, photons, ron, start_after_iteration = start_after_iteration, modulation = 0)
    
    WFSParams_Pyramid = Config.fromfile(paramfile)['WFSParams']
    WFSParams_Pyramid["Reconstruction"] = "Linear"
    WFSParams_Pyramid["MaskType"] = "Pyramid"
    WFSParams_Pyramid["Modulation"] = 5
    WFSParams_Pyramid["InitParam"] = [1.57, 1.57]
    
    Pyramid_End2EndWFS_mod5 = End2EndWFS (WFSParams_Pyramid,AtmosParams,device)
    Pyramid_End2EndWFS_mod5.eval()
    Pyramid_AO_Loop_mod5 = AOLoop(WFSParams_Pyramid, Pyramid_End2EndWFS_mod5, z_FullRes, gain, leak, phaseGT, zernike, photons, ron, start_after_iteration = start_after_iteration, modulation = 0)
        
    
    perfect_residual_variance = torch.var(phaseGT[:, dataset.pupil.bool()], dim=-1).unsqueeze(-1)
    input_variance = torch.var(phaseGT[:, dataset.pupil.bool()], dim=-1).unsqueeze(-1)
    recon = torch.zeros_like(phaseGT)
    perfect_residual = phaseGT - recon
        
    
    # Pyramid_End2EndWFS_mod0.WFS.reconstructionMatrix = Pyramid_End2EndWFS_mod5.WFS.reconstructionMatrix
    
#%%    
    
    
    
    
    with torch.no_grad():
    
        
        
        sample_to_look = 0
        
        fig, ax = plt.subplots(1, 7, figsize=(20, 5))
        img1 = ax[0].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img2 = ax[1].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img3 = ax[2].imshow(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
        img4 = ax[3].imshow(Trained_AO_Loop.images[sample_to_look,:,:].cpu().detach().numpy())
        img5 = ax[4].imshow(Trained_AO_Loop.End2EndWFS.WFS.psf_no_noise[sample_to_look,0,:,:].cpu().detach().numpy())
        img6 = ax[5].imshow(Trained_AO_Loop.long_exposure_psf[sample_to_look,0,:,:].cpu().detach().numpy())
        plot4, = ax[6].semilogy(zernike[sample_to_look,:].cpu().detach().numpy())
        plot5, = ax[6].semilogy(zernike[sample_to_look,:].cpu().detach().numpy())
        plot6, = ax[6].semilogy(zernike[sample_to_look,:].cpu().detach().numpy())
        plot7, = ax[6].semilogy(zernike[sample_to_look,:].cpu().detach().numpy(), 'k--')
        plot8, = ax[6].semilogy(zernike[sample_to_look,:].cpu().detach().numpy(), 'r--')
        plt.subplots_adjust(left=0.02, right=0.98, wspace=0.1)
        # plt.subplots_adjust(left = 0.1, top = 0.9, right = 0.9, bottom = 0.1, hspace = 0.5, wspace = 0.5)


        
        for i in range(500):
            # Get new WFS images after applying the correction
            phaseGT,zernike,_,_,_,_,_ = dataset.GetMovingWavefront()
            
            Trained_AO_Loop.step(phaseGT)
            AO_Loop_Linear.step(phaseGT)
            Pyramid_AO_Loop_mod5.step(phaseGT)

            
            perfect_residual = phaseGT - recon
            Ze = torch.matmul(perfect_residual.flatten(1,2), dataset.invZ)# * Trained_AO_Loop.filter
            recon = recon*leak + (Ze @ z_FullRes).view_as(phaseGT) * gain
            perfect_residual_variance = torch.cat((perfect_residual_variance, torch.var(perfect_residual[:, dataset.pupil.bool()], dim = -1).unsqueeze(-1)), dim = 1)
            input_variance = torch.cat((input_variance, torch.var(phaseGT[:, dataset.pupil.bool()], dim = -1).unsqueeze(-1)), dim = 1)
            
            
            if i % 20 == 0:
                plt.subplot(1,7,1)
                plt.title(f'Input phase r0 = {r0s[sample_to_look,:].item():.3f}')
                img1.set_data(phaseGT[sample_to_look,:,:].cpu().detach().numpy())
                img1.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
                
                
                plt.subplot(1,7,2)
                plt.title('Reconstructed phase')
                img2.set_data(Trained_AO_Loop.z_reconstructed[sample_to_look,:,:].cpu().detach().numpy())
                img2.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
                
                plt.subplot(1,7,3)
                plt.title('Residual phase')
                img3.set_data(Trained_AO_Loop.residual_phase[sample_to_look,:,:].cpu().detach().numpy())
                img3.set_clim(vmin=np.min(img1.get_array()), vmax=np.max(img1.get_array()))
                
                plt.subplot(1,7,4)
                plt.title(f'Photons = {photons[sample_to_look,:,:].item():.0f}, RON = {ron[sample_to_look,:,:].item():.2f}')
                img4.set_data(Trained_AO_Loop.images[sample_to_look,:,:].cpu().detach().numpy())
                img4.set_clim(vmin=np.min(img4.get_array()), vmax=np.max(img4.get_array()))
                
                plt.subplot(1,7,5)
                plt.title('Short exposure PSF')
                img5.set_data((Trained_AO_Loop.End2EndWFS.WFS.psf_no_noise[sample_to_look,0,:,:]).cpu().detach().sqrt().numpy())
                img5.set_clim(vmin=np.min(img5.get_array()), vmax=np.max(img5.get_array()))
                
                
                plt.subplot(1, 7, 6)
                plt.title('Long exposure PSF')
                img6.set_data((Trained_AO_Loop.long_exposure_psf[sample_to_look,0,:,:]).cpu().detach().sqrt().numpy())
                img6.set_clim(vmin=np.min(img6.get_array()), vmax=np.max(img6.get_array()))
                
                
                plt.subplot(1, 7, 7)
                plt.title("Residual variance")
                plot4.set_data(np.arange(Trained_AO_Loop.residual_variance.shape[1]), Trained_AO_Loop.residual_variance[sample_to_look,:].cpu().detach().numpy())
                plot5.set_data(np.arange(Pyramid_AO_Loop_mod5.residual_variance.shape[1]), Pyramid_AO_Loop_mod5.residual_variance[sample_to_look,:].cpu().detach().numpy())
                plot6.set_data(np.arange(AO_Loop_Linear.residual_variance.shape[1]), AO_Loop_Linear.residual_variance[sample_to_look,:].cpu().detach().numpy())
                plot7.set_data(np.arange(AO_Loop_Linear.residual_variance.shape[1]), perfect_residual_variance[sample_to_look,:].cpu().detach().numpy())
                plot8.set_data(np.arange(AO_Loop_Linear.residual_variance.shape[1]), input_variance[sample_to_look,:].cpu().detach().numpy())
                
                ax[6].relim()
                ax[6].autoscale_view()
                plt.xlabel('Iteration')
                plt.ylabel('Residual variance', labelpad=10)
                plt.legend(['NN', 'Pyramid mod 5', 'Linear VZernike', 'Perfect reconstruction', 'Input variance'])
                plt.pause(0.1)
                
                fig.canvas.draw()
            
            # if i % 2 == 0:
            #     image = np.array(fig.canvas.buffer_rgba())
            #     writer.append_data(image)
            
            
            if i % 50 == 0:
               sample_to_look += 1
            
        # writer.close()
        
