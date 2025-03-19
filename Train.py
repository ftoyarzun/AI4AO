#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:24:10 2025

@author: ptrouve
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
    
class Custom_Loss_Function(nn.Module):
    def __init__(self, epsilon = 1e-2, degree = 2, NZernike = 50, device = 'cuda'):
        super().__init__()
        self.epsilon = epsilon
        self.degree = degree

        self.linspace = torch.sqrt((torch.linspace(1, NZernike, NZernike, device=device)))
        
    def forward(self, y_pred, y_true, r0):
        diff = (y_pred - y_true) * self.linspace
        return torch.mean(torch.abs(diff) ** self.degree * r0)



def test (End2EndWFS, dataset, loss, TestRunNb, device = 'cuda'):
    

    
    wfs = End2EndWFS.WFSmodule.WFS
    
    wfs_Zernike = copy.copy(wfs)
    wfs_Zernike.param = [2, 1.57, 100]
    wfs_Zernike.BuildZernikeMask()
    
    wfs_Pyramid = copy.copy(wfs)
    wfs_Pyramid.param = [0.78, 0.78]
    wfs_Pyramid.BuildPyramidMask()
    
    Nzernike = dataset.Nzernike
    
    [z, z_FullRes] = Zernike(wfs.pupil, wfs.pupil_logical, wfs.Nres, Nzernike)
    
    wfs_Zernike.BuildReconstructionMatrix(z_FullRes, wfs.mask)   
    wfs_Pyramid.BuildReconstructionMatrix(z_FullRes, wfs.mask)
            
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
    
    with torch.no_grad():
        
        End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons,ron)
        output = End2EndWFS(phaseGT)    
        
        wfs_Zernike.SetPhotonsAndRON(photons,ron)
        test_frame_wfs_Zernike = wfs_Zernike.Propagator(phaseGT)
        estimated_phase_wfs_Zernike = wfs_Zernike.GetReconstructedPhase(test_frame_wfs_Zernike)
        
        wfs_Pyramid.SetPhotonsAndRON(photons,ron)
        test_frame_wfs_Pyramid = wfs_Pyramid.Propagator(phaseGT)
        estimated_phase_wfs_Pyramid = wfs_Pyramid.GetReconstructedPhase(test_frame_wfs_Pyramid)
        

        loss_NN = loss(output,zernike.to(output.dtype), r0s)
        loss_Zernike = loss(estimated_phase_wfs_Zernike,zernike.to(estimated_phase_wfs_Zernike.dtype), r0s)
        loss_Pyramid = loss(estimated_phase_wfs_Pyramid,zernike.to(estimated_phase_wfs_Pyramid.dtype), r0s)
        
        
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


def log_scale_zernike(zernike):
    # return torch.sign(zernike) * torch.log(torch.abs(zernike))
    
    # Get the length of the Zernike modes
    length = zernike.shape[-1]
    
    # Create a linspace from 1 to the length
    linspace = torch.linspace(1, length, length, device=zernike.device)  # Make sure the device matches
    
    # Scale the Zernike modes with the linspace
    
    return zernike * torch.sqrt(linspace)




def train (End2EndWFS, dataset, loss, TrainRunNb, optimizer_o,optimizer_n, device = 'cuda'):
    
    final_train_loss = 0
    
    loss_tracker = []
    param_tracker = []
    
    fig,ax = plt.subplots() 
    img = ax.imshow(End2EndWFS.maskGenerator(End2EndWFS.uv_coords).view(End2EndWFS.N,End2EndWFS.N).cpu().detach().numpy())
    fig.colorbar(img)
    
    # writer = imageio.get_writer(gif_path, mode="i", fps = 10)
    
    End2EndWFS.train()
    
    for u in range(0,TrainRunNb) :
        

        
        optimizer_n.zero_grad()
        optimizer_o.zero_grad()
        
        phaseGT,zernike,photons,ron,r0s = dataset[0]
        
        End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons,ron)
        
        output = End2EndWFS(phaseGT)  
        
   
        # take a batch of images to estimage a batch of zernike parameters
     
        l = loss(output,zernike.to(output.dtype),r0s)
   
          
        l. backward()
        
        #Comment one of these line to stop optimization on either part
        
        optimizer_o.step()
        optimizer_n.step()
        
        # with torch.no_grad():  # Clip the parameter values after optimization step
        
        #     for param in End2EndWFS.WFSmodule.WFS.param:
        #         param.clamp_(0.001, 1000)
        
        

        # parameters values should change during the loop
        #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  param_proc  : {:.7f}\n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item(),Trained_End2EndWFS.PhaseEstimator.param[0,0].item()), end="")
        if u % 100 == 0:
            #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {u}, train loss : {l.item():.5f} param_opt {Trained_End2EndWFS.WFSmodule.WFS.param.tolist()}")
            loss_tracker.append(l.item())
            param_tracker.append(Trained_End2EndWFS.WFSmodule.WFS.param.tolist())
            img.set_data(End2EndWFS.maskGenerator(End2EndWFS.uv_coords).view(End2EndWFS.N,End2EndWFS.N).cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            
            fig.canvas.draw()
            # image = np.array(fig.canvas.buffer_rgba())
            # writer.append_data(image)
            
            plt.pause(0.1)
        
        final_train_loss = l +final_train_loss
        # writer.close()
    return loss_tracker, np.array(param_tracker)



def train_closed_loop_static(End2EndWFS, dataset, loss, TrainRunNb, optimizer_o, optimizer_n, gain=0.3, num_iterations=5, device='cuda'):

    loss_tracker = []
    param_tracker = []
    
    fig, ax = plt.subplots()
    img = ax.imshow(End2EndWFS.maskGenerator(End2EndWFS.uv_coords).view(End2EndWFS.N, End2EndWFS.N).cpu().detach().numpy())
    fig.colorbar(img)
    
    z_FullRes = dataset.z_FullRes.to(device, dtype=torch.float32).view(-1, 50).transpose(0, 1)

    End2EndWFS.train()

    for u in range(TrainRunNb):
        #print(u)
        optimizer_n.zero_grad()
        optimizer_o.zero_grad()

        # Get dataset sample
        phaseGT, zernike, photons, ron, r0s = dataset[0]
        End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons, ron)

        # **Closed-loop iterative correction**
        z_estimated = torch.zeros_like(zernike)  # Start with zero correction
        residual_phase = phaseGT.clone()  # Start with the original phase

        total_loss = 0
        for i in range(num_iterations):
            # Get new WFS images after applying the correction
            

            # Predict Zernike coefficients and update estimate
            z_output = End2EndWFS(residual_phase)
            z_estimated = z_estimated + gain * z_output  # Apply correction with gain

            # Convert Zernike coefficients to full-resolution wavefront
            z_reconstructed = torch.matmul(z_estimated, z_FullRes).view_as(phaseGT)

            # Compute residual phase after correction
            residual_phase = phaseGT - z_reconstructed

            # Compute loss for this iteration
            iter_loss = loss(z_estimated, zernike.to(z_estimated.dtype), r0s)
            total_loss += iter_loss

        # **Backpropagation**
        total_loss.backward()
        optimizer_o.step()
        optimizer_n.step()

        # **Track loss and parameters**
        if u % 100 == 0:
            print(f"Run {u}, Train Loss: {total_loss.item():.5f}, Params: {End2EndWFS.WFSmodule.WFS.param.tolist()}")
            loss_tracker.append(total_loss.item())
            param_tracker.append(End2EndWFS.WFSmodule.WFS.param.tolist())

            img.set_data(End2EndWFS.maskGenerator(End2EndWFS.uv_coords).view(End2EndWFS.N, End2EndWFS.N).cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            fig.canvas.draw()
            plt.pause(0.1)



    return loss_tracker, np.array(param_tracker)


def train_closed_loop_dynamic(End2EndWFS, dataset, loss, TrainRunNb, optimizer_o, optimizer_n, gain=0.3, num_iterations=5, device='cuda'):

    loss_tracker = []
    param_tracker = []
    
    fig, ax = plt.subplots()
    img = ax.imshow(End2EndWFS.maskGenerator(End2EndWFS.uv_coords).view(End2EndWFS.N, End2EndWFS.N).cpu().detach().numpy())
    fig.colorbar(img)
    
    z_FullRes = dataset.z_FullRes.to(device, dtype=torch.float32).view(-1, 50).transpose(0, 1)

    End2EndWFS.train()

    for u in range(TrainRunNb):
        #print(u)
        optimizer_n.zero_grad()
        optimizer_o.zero_grad()

        # Get dataset sample
        dataset.ResetMovingWavefront()
        phaseGT,zernike,photons,ron,r0s,_,_ = dataset.GetMovingWavefront(3, 0.001)
        End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons, ron)

        # **Closed-loop iterative correction**
        z_estimated = torch.zeros_like(zernike)  # Start with zero correction
        
        z_reconstructed = torch.zeros_like(phaseGT)
        residual_phase = torch.zeros_like(phaseGT)  # Start with the original phase

        total_loss = 0
        for i in range(num_iterations):
            # Get new WFS images after applying the correction
            phaseGT,zernike,_,_,_,_,_ = dataset.GetMovingWavefront(3, 0.001)
            residual_phase = phaseGT - z_reconstructed
            # Predict Zernike coefficients and update estimate
            z_output = End2EndWFS(residual_phase)
            z_estimated = z_estimated + gain * z_output  # Apply correction with gain

            # Convert Zernike coefficients to full-resolution wavefront
            z_reconstructed = torch.matmul(z_estimated, z_FullRes).view_as(phaseGT)

            # Compute residual phase after correction
            

            # Compute loss for this iteration
            if i > num_iterations - 4:
                iter_loss = loss(z_estimated, zernike.to(z_estimated.dtype), r0s)
                total_loss += iter_loss

        # **Backpropagation**
        total_loss.backward()
        optimizer_o.step()
        optimizer_n.step()

        # **Track loss and parameters**
        if u % 10 == 0:
            print(f"Run {u}, Train Loss: {total_loss.item():.5f}, Params: {End2EndWFS.WFSmodule.WFS.param.tolist()}")
            loss_tracker.append(total_loss.item())
            param_tracker.append(End2EndWFS.WFSmodule.WFS.param.tolist())

            img.set_data(End2EndWFS.maskGenerator(End2EndWFS.uv_coords).view(End2EndWFS.N, End2EndWFS.N).cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            fig.canvas.draw()
            plt.pause(0.1)



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
    dataset = PhaseDataset(WFSParams['D'],WFSParams['Nres'],WFSParams['Nzernike'],                       
                           AtmosParams['L0'],AtmosParams['r0'],AtmosParams['Nphases'],
                           WFSParams['Nactuator'],LoopParams['levelOfCorrection'],
                           LoopParams['loopFrequency'], LoopParams['delayFrames'], LoopParams['windSpeedVector'],
                           WFSParams['Nphotons'], WFSParams['RON'], device)
    
    
    # Setting the loss function
    loss = Custom_Loss_Function()
    
    # Initialisation of the system 
    Trained_End2EndWFS = End2EndWFS (WFSParams,device)
    
    
   
    
    # Optimization parameters (learning rate lr and nb of runs)
    lrn = TrainParams['lrn']
    lro = TrainParams['lro']
    
    # Number of training and testing run 
    
    TrainRunNb = TrainParams['TrainRunNb']
    TestRunNb = TrainParams['TestRunNb']
    
    # Setting the optimizer (here Adam)
    
    # To optimize the optical and the processing parameters
    if WFSParams['MaskType'] != "Free":
        optimizer_o = torch.optim.AdamW([Trained_End2EndWFS.WFSmodule.WFS.param],lro)
    else:
        optimizer_o = torch.optim.AdamW(Trained_End2EndWFS.maskGenerator.parameters(),lro)
   
    
   
    optimizer_n = torch.optim.AdamW(Trained_End2EndWFS.PhaseEstimator.parameters(),lrn)
 
   
    # Training part for parameters optimization
    print("Initialized parameters",Trained_End2EndWFS.WFSmodule.param)
    
    
    checkpoint_path = 'FreeTest4.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        Trained_End2EndWFS.load_state_dict(checkpoint['model_state_dict'])
        optimizer_o.load_state_dict(checkpoint['optimizer_o_state_dict'])
        optimizer_n.load_state_dict(checkpoint['optimizer_n_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Resuming from epoch", start_epoch)
    else:
        start_epoch = 0
        print("Starting from scratch")
    
    
    a = time.time()
    train_loss, train_parameters = train_closed_loop_dynamic(Trained_End2EndWFS,dataset,loss,TrainRunNb,optimizer_o,optimizer_n, 0.7, 10, device)
    # train_loss, train_parameters = train_closed_loop_static(Trained_End2EndWFS,dataset,loss,TrainRunNb,optimizer_o,optimizer_n, 1., 7, device)
    # train_loss, train_parameters = train(Trained_End2EndWFS,dataset,loss,TrainRunNb,optimizer_o,optimizer_n,device)
    b = time.time() - a 
    
    # Testing part
    permanentPhaseDataset = PermanentPhaseDataset()
    test_loss = test(Trained_End2EndWFS,permanentPhaseDataset,loss,TestRunNb,device)
    
    # Save network and parameters
    
    torch.save({
        'epoch': start_epoch + TrainParams['TrainRunNb'],  # Update the epoch number after training
        'model_state_dict': Trained_End2EndWFS.state_dict(),
        'optimizer_o_state_dict': optimizer_o.state_dict(),
        'optimizer_n_state_dict': optimizer_n.state_dict(),
    }, checkpoint_path)
    
    
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
        