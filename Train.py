#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:24:10 2025

@author: ptrouve
"""
import torch

from PhaseDataset import PhaseDataset, PermanentPhaseDataset
from SimuEnd2EndWFS import End2EndWFS, CheckpointManager
from LossFunctions import Custom_Loss_Function, Physics_loss, ResidualPhaseLoss
from MaskGeneration import MaskVisualizator
from mmengine import Config
import numpy as np
import matplotlib.pyplot as plt
import time
from TorchPropagator import Zernike
import copy
from collections import defaultdict

import imageio.v2 as imageio

from line_profiler import profile
    

    
def test (End2EndWFS, dataset, loss, TestRunNb, device = 'cuda'):
    

    
    wfs = End2EndWFS.WFSmodule.WFS
    
    wfs_Zernike = copy.copy(wfs)
    wfs_Zernike.param = [2, 1.57, 100]
    wfs_Zernike.BuildZernikeMask()
    
    wfs_Pyramid = copy.copy(wfs)
    wfs_Pyramid.param = [0.78, 0.78]
    wfs_Pyramid.BuildPyramidMask()
    
    Nzernike = dataset.Nzernike
    
    z_FullRes = dataset.z_FullRes.permute(2,0,1).to(device = device)
    
    wfs_Zernike.BuildReconstructionMatrix(z_FullRes)   
    wfs_Pyramid.BuildReconstructionMatrix(z_FullRes)
            
    wfs_Zernike.BuildReferenceIntensity()
    wfs_Pyramid.BuildReferenceIntensity()
  
    phaseGT,zernike,photons,ron,r0s =dataset[0]
    End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons,ron)
      
    plt.close('all')   
    output = End2EndWFS(phaseGT[0,:,:]) 
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
 
    
 
    img = axes[0].imshow(End2EndWFS.WFSmodule(phaseGT)[0,:,:].cpu().detach().numpy())
    line1, = axes[1].plot(zernike[0,:].cpu().data.numpy())
    line2, = axes[1].plot(output[0,:].cpu().data.numpy())
    line3, = axes[1].plot(output[0,:].cpu().data.numpy())
    line4, = axes[1].plot(output[0,:].cpu().data.numpy())
    
    line5, = axes[2].plot(output[0,:].cpu().data.numpy())
    line6, = axes[2].plot(output[0,:].cpu().data.numpy())
    line7, = axes[2].plot(output[0,:].cpu().data.numpy())
    
    
    axes[1].set_ylim(-1, 1)
    axes[2].set_ylim(-1, 1)
    
    final_test_loss = 0
    
    End2EndWFS.eval()
    # End2EndWFS.train()
    
    with torch.no_grad():
        
        End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons,ron)
        output = End2EndWFS(phaseGT)    
        
        wfs_Zernike.SetPhotonsAndRON(photons,ron)
        test_frame_wfs_Zernike = wfs_Zernike.Propagator(phaseGT)
        estimated_phase_wfs_Zernike = wfs_Zernike.GetReconstructedPhase(test_frame_wfs_Zernike)
        
        wfs_Pyramid.SetPhotonsAndRON(photons,ron)
        test_frame_wfs_Pyramid = wfs_Pyramid.Propagator(phaseGT)
        estimated_phase_wfs_Pyramid = wfs_Pyramid.GetReconstructedPhase(test_frame_wfs_Pyramid)
        

        loss_NN = loss(output,zernike, r0s)
        loss_Zernike = loss(estimated_phase_wfs_Zernike,zernike, r0s)
        loss_Pyramid = loss(estimated_phase_wfs_Pyramid,zernike, r0s)
        
        
        final_test_loss = loss_NN +final_test_loss
        
        # print(" Test Run n°  {}, test loss : {:.4f}  \n".format(u, loss_NN), end="")
        print(f'NN loss = {loss_NN.item():.4f}, Zernike WFS loss = {loss_Zernike.item():.4f}, Pyramid WFS loss = {loss_Pyramid.item():.4f}')
        
    
        for u in range(10) :
 
            #plot of an example within the batch
            
            plt.subplot(1,3,1)
            plt.title('WFS image')
            img.set_data(End2EndWFS.WFSmodule(phaseGT)[u,:,:].cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            #plt.colorbar()
            #plt.draw()
        
            plt.subplot(1,3,2)
            plt.title('Phase estimation')
            line1.set_ydata(zernike[u,:].cpu().data.numpy())
            line2.set_ydata(output[u,:].cpu().data.numpy())
            line3.set_ydata(estimated_phase_wfs_Zernike[...,u,:].cpu().data.numpy())
            line4.set_ydata(estimated_phase_wfs_Pyramid[...,u,:].cpu().data.numpy())
            plt.xlabel('Zernike mode index')
            plt.ylabel('Zernike mode Amplitude')
            plt.title(f'NN loss = {loss_NN.item():.4f}, Zernike WFS loss = {loss_Zernike.item():.4f}, Pyramid WFS loss = {loss_Pyramid.item():.4f}')
            plt.legend(['Ground truth', 'CNN', 'Linear reconctructor Zernike WFS', 'Linear reconstructor Pyramid WFS'])
            
            plt.subplot(1,3,3)
            plt.title('Phase residuals')
            line5.set_ydata(zernike[u,:].cpu().data.numpy() - output[u,:].cpu().data.numpy())
            line6.set_ydata(zernike[u,:].cpu().data.numpy() - estimated_phase_wfs_Zernike[...,u,:].cpu().data.numpy())
            line7.set_ydata(zernike[u,:].cpu().data.numpy() - estimated_phase_wfs_Pyramid[...,u,:].cpu().data.numpy())
            plt.xlabel('Zernike mode index')
            plt.ylabel('Zernike mode Amplitude')
            plt.legend(['CNN', 'Linear reconctructor Zernike WFS', 'Linear reconstructor Pyramid WFS'])
            
            plt.show()
            
            plt.pause(2)
            
            
   
    return final_test_loss /TestRunNb





def train_closed_loop(End2EndWFS, dataset, loss, TrainRunNb, optimizer_o, optimizer_n, gain=0.3, num_iterations=5, device='cuda'):

    loss_tracker = []
    param_tracker = []
    
    # fig, ax = plt.subplots()
    # img = ax.imshow(End2EndWFS.phaseMask.cpu().detach().numpy())
    # fig.colorbar(img)
    maskVisualizator.SetCanvas()
    
    z_FullRes = dataset.z_FullRes.to(device, dtype=torch.float32).view(-1, dataset.Nzernike).transpose(0, 1)

    End2EndWFS.train()

    for u in range(TrainRunNb // num_iterations):
        t1 = time.perf_counter()
        # Get dataset sample
        
        dataset.ResetMovingWavefront()
        phaseGT,zernike,photons,ron,r0s,_,_ = dataset.GetMovingWavefront(generateClosedLoop = (num_iterations == 1))
        End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons, ron)
        
        

        # **Closed-loop iterative correction**
        z_estimated = torch.zeros_like(zernike)  # Start with zero correction        
        z_reconstructed = torch.zeros_like(phaseGT)

        total_loss = 0
        for i in range(num_iterations):
            # Get new WFS images after applying the correction
            if i > 0:
                phaseGT,zernike,_,_,_,_,_ = dataset.GetMovingWavefront(generateClosedLoop = (num_iterations == 1))
            residual_phase = phaseGT - z_reconstructed

            Ze = torch.matmul(residual_phase.flatten(1,2), dataset.invZ)
            # Predict Zernike coefficients and update estimate
           
            z_output = End2EndWFS(residual_phase)
            
            z_estimated = z_estimated + gain * z_output  # Apply correction with gain

            
            # Convert Zernike coefficients to full-resolution wavefront
            if num_iterations > 1:
                z_reconstructed = torch.matmul(z_estimated, z_FullRes).view_as(phaseGT)
            
            # Compute loss for this iteration

            convergence_ratio = 1 - (1 - gain) ** (i + 1)

            # iter_loss = loss(z_reconstructed, phaseGT * convergence_ratio, r0s)
            # iter_loss1 = loss[1](z_estimated, zernike * convergence_ratio, r0s)
            # iter_loss0 = loss[1](z_output, Ze * gain, r0s) * 1
            iter_loss1 = loss[2](z_estimated, phaseGT * convergence_ratio, r0s) * 2
            # iter_loss0 = loss[0](End2EndWFS.WFSmodule, End2EndWFS.Image, z_estimated, r0s) * 1
            # total_loss += (iter_loss0 + iter_loss1) / num_iterations
            total_loss = iter_loss1
            
            
            
        # **Backpropagation**
        optimizer_n.state = defaultdict(dict)
        optimizer_o.state = defaultdict(dict)
        
        
        optimizer_n.zero_grad(set_to_none = True)
        optimizer_o.zero_grad(set_to_none = True)
        
        
        total_loss.backward()
        optimizer_o.step()
        optimizer_n.step()
        
        t2 = time.perf_counter()
        # **Track loss and parameters**
        if u % 10 == 0:
            #t2 = time.perf_counter()
            print(f'Iteration frequency = {1/(t2 - t1)}')
            # print(f"Run {u}, Train Loss Image: {iter_loss0.item():.10f}, Train Loss Zernike: {iter_loss1.item():.10f}, Params: {End2EndWFS.WFSmodule.WFS.param.tolist()}")
            print(f"Run {u}, Train Loss Zernike: {iter_loss1.item():.10f}")
            loss_tracker.append(total_loss.item())
            # param_tracker.append(End2EndWFS.WFSmodule.WFS.param.tolist())

            # img.set_data(End2EndWFS.phaseMask.cpu().detach().numpy())
            # img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            # # fig.canvas.draw()
            # #t1 = time.perf_counter()
            # plt.pause(0.1)
            maskVisualizator.show()
            


    return loss_tracker, np.array(param_tracker)


if __name__ == "__main__":
    
    device = 'cuda' # set to "cpu" if Cuda is not available
    
    paramfile = 'params_exp.py'  # file of experimental parameters

    gif_path = 'free_mask_animation_2ndStage3.gif'

    # Config extraction
    AtmosParams = Config.fromfile(paramfile)['AtmosParams']
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']
    TrainParams = Config.fromfile(paramfile)['TrainParams']

    # Dataset creation
    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, device)
    z_FullRes = dataset.z_FullRes.to(device, dtype=torch.float32).view(-1, dataset.Nzernike).transpose(0, 1)
    

    # Initialisation of the system 
    Trained_End2EndWFS = End2EndWFS (WFSParams,device, "SimpleNet")
    
    
    # Setting the loss function
    loss_test = Custom_Loss_Function(NZernike = WFSParams["Nzernike"])
    loss_residual = ResidualPhaseLoss(z_FullRes, dataset.pupil)
    loss_physics = Physics_loss(z_FullRes, dataset[0][0])
    loss = [loss_physics, loss_test, loss_residual]
    
   
    
    # Optimization parameters (learning rate lr and nb of runs)
    lrn = TrainParams['lrn']
    lro = TrainParams['lro']
    
    # Number of training and testing run 
    
    TrainRunNb = TrainParams['TrainRunNb']
    TestRunNb = TrainParams['TestRunNb']
    
    # Setting the optimizer (here Adam)
    
    # To optimize the optical and the processing parameters
    if WFSParams['MaskType'] == "Pyramid" or WFSParams['MaskType'] == "Zernike":
        optimizer_o = torch.optim.AdamW([Trained_End2EndWFS.WFSmodule.WFS.param],lro)
    elif WFSParams['MaskType'] == "FreePhase":
        optimizer_o = torch.optim.AdamW([
                                        {'params': Trained_End2EndWFS.phaseMaskGenerator.parameters(), 'lr': lro, 'fused': True}
                                        ])
    elif WFSParams['MaskType'] == "FreeTransmision":
        optimizer_o = torch.optim.AdamW([
                                        {'params': Trained_End2EndWFS.transmisionMaskGenerator.parameters(), 'lr': lro, 'fused': True}
                                        ])
    elif WFSParams['MaskType'] == "FreePhaseTransmision":
        optimizer_o = torch.optim.AdamW([
                                        {'params': Trained_End2EndWFS.phaseMaskGenerator.parameters(), 'lr': lro},
                                        {'params': Trained_End2EndWFS.transmisionMaskGenerator.parameters(), 'lr': lro, 'fused': True}
                                        ])
    
    optimizer_n = torch.optim.AdamW(Trained_End2EndWFS.PhaseEstimator.parameters(),lrn, fused = True)
 
    checkpoint_path = 'PhaseTransmisionTest2.pth'
    checkpointManager = CheckpointManager(Trained_End2EndWFS, WFSParams, TrainParams, checkpoint_path, optimizer_o, optimizer_n)
    
    checkpointManager.load()
    # checkpointManager.load_network()
    # checkpointManager.load_free_phaseMask('PhaseTransmisionTest2.pth')
    # checkpointManager.load_model()

   
    # Training part for parameters optimization
    print("Initialized parameters",Trained_End2EndWFS.WFSmodule.param)

    
    maskVisualizator = MaskVisualizator(Trained_End2EndWFS)
    
    
    a = time.time()
    
    train_loss, train_parameters = train_closed_loop(Trained_End2EndWFS,
                                                     dataset,
                                                     loss,
                                                     TrainRunNb,
                                                     optimizer_o,
                                                     optimizer_n,
                                                     gain=0.5, 
                                                     num_iterations=30,
                                                     device=device)
    
    
    b = time.time() - a 
    
    # Save network and parameters
    checkpointManager.save()
    
    
    # Testing part
    permanentPhaseDataset = PermanentPhaseDataset()
    
    var_loss = ResidualPhaseLoss(dataset.z_FullRes, dataset.pupil)
    
    test_loss = test(Trained_End2EndWFS,dataset,loss_test,TestRunNb,device)
    
    
    
    
    
    
    plt.subplots(1, 3, figsize=(20, 5))
    plt.subplot(1,3,1)
    plt.semilogy(train_loss)
    plt.title('Loss evolution')
    plt.xlabel('Iteration')
    plt.ylabel('MSE loss')
    
    plt.subplot(1,3,2)
    plt.plot(train_parameters[:,0])
    plt.title('Diameter evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Dot diameter in lambda/D')

    plt.subplot(1,3,3)
    plt.plot(train_parameters[:,1])  
    plt.title('Depth evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Dot depth in rad')
        