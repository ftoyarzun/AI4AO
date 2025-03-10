#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:24:10 2025

@author: ptrouve
"""
import torch

from PhaseDataset import PhaseDataset
from SimuEnd2EndWFS import End2EndWFS
from mmengine import Config
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from TorchPropagator import Zernike
import copy

def test (End2EndWFS, dataset, loss, TestRunNb, device = 'cuda'):
    

    
    wfs = End2EndWFS.WFSmodule.WFS
    wfs.photonRange = [4 ,4]
    wfs.RONRange = [2, 2]
    
    
    wfs_Zernike = copy.copy(wfs)
    wfs_Zernike.param = [2, 1.57, 100]
    wfs_Zernike.BuildZernikeMask()
    
    wfs_Pyramid = copy.copy(wfs)
    wfs_Pyramid.param = [0.78, 0.78]
    wfs_Pyramid.BuildPyramidMask()
    
    Nzernike = dataset.Nzernike
    
    [z, z_FullRes] = Zernike(wfs.pupil, wfs.pupil_logical, wfs.Nres, Nzernike)
    
    wfs_Zernike.BuildReconstructionMatrix(torch.tensor(z_FullRes, dtype=torch.float32, device=wfs.mask.device), wfs.mask)
    wfs_Pyramid.BuildReconstructionMatrix(torch.tensor(z_FullRes, dtype=torch.float32, device=wfs.mask.device), wfs.mask)
    
    wfs_Zernike.BuildReferenceIntensity()
    wfs_Pyramid.BuildReferenceIntensity()
  
    phaseGT,zernike,_,_=dataset[0]
      
    plt.close('all')   
    output = End2EndWFS(phaseGT[0,:,:]) 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    img = axes[0].imshow(End2EndWFS.WFSmodule(phaseGT)[0,:,:].cpu().detach().numpy())
    line1, = axes[1].plot(zernike[0,:].cpu().data.numpy())
    line2, = axes[1].plot(output[0,:].cpu().data.numpy())
    line3, = axes[1].plot(output[0,:].cpu().data.numpy())
    line4, = axes[1].plot(output[0,:].cpu().data.numpy())
    
    axes[1].set_ylim(-0.1, 0.1)
    
    final_test_loss = 0
    
    End2EndWFS.eval()
    
    with torch.no_grad():
    
        for u in range(1) :
 
            phaseGT,zernike,_,_=dataset[0]
            
           
          
           # take a batch of images to estimage a batch of zernike parameters
            output = End2EndWFS(phaseGT)  
            
            test_frame_wfs_Zernike = wfs_Zernike.Propagator(phaseGT[0,:,:])
            estimated_phase_wfs_Zernike = wfs_Zernike.GetReconstructedPhase(test_frame_wfs_Zernike)
            
            
            test_frame_wfs_Pyramid = wfs_Pyramid.Propagator(phaseGT[0,:,:])
            estimated_phase_wfs_Pyramid = wfs_Pyramid.GetReconstructedPhase(test_frame_wfs_Pyramid)
            
    
            l = loss(output,zernike.to(output.dtype))
            
            
            final_test_loss = l +final_test_loss
            
            print(" Test Run n°  {}, test loss : {:.4f}  \n".format(u, l), end="")
            
            
            #plot of an example within the batch
            
            plt.subplot(1,2,1)
            plt.title('WFS image')
            img.set_data(End2EndWFS.WFSmodule(phaseGT)[0,:,:].cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            #plt.colorbar()
            #plt.draw()
        
            plt.subplot(1,2,2)
            plt.title('Phase estimation')
            line1.set_ydata(zernike[0,:].cpu().data.numpy())
            line2.set_ydata(output[0,:].cpu().data.numpy())
            line3.set_ydata(estimated_phase_wfs_Zernike[0,:].cpu().data.numpy())
            line4.set_ydata(estimated_phase_wfs_Pyramid[0,:].cpu().data.numpy())
            plt.xlabel('Zernike mode index')
            plt.ylabel('Zernike mode Amplitude')
            plt.title(" MSE " + str(loss(output[0,:],zernike[0,:]).item()))
            plt.legend(['Ground truth', 'CNN', 'Linear reconctructor Zernike WFS', 'Linear reconstructor Pyramid WFS'])
            plt.show()
            
            plt.pause(0.1)
            
            
   
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
    
    End2EndWFS.train()
    
    for u in range(0,TrainRunNb) :
        
        
        
        
        optimizer_n.zero_grad()
        optimizer_o.zero_grad()
        
        phaseGT,zernike,_,_=dataset[0]
        
              
        output = End2EndWFS(phaseGT)  
        
   
        # take a batch of images to estimage a batch of zernike parameters
     
        l = loss(log_scale_zernike(output),log_scale_zernike(zernike.to(output.dtype)))
   
          
        l. backward()
        
        #Comment one of these line to stop optimization on either part
        
        optimizer_o.step()
        optimizer_n.step()
        
        # with torch.no_grad():  # Clip the parameter values after optimization step
        
        #     for param in End2EndWFS.WFSmodule.WFS.param:
        #         param.clamp_(0.001, 1000)
        
        

        # parameters values should change during the loop
        #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  param_proc  : {:.7f}\n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item(),Trained_End2EndWFS.PhaseEstimator.param[0,0].item()), end="")
        if u % 10 == 0:
            #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {u}, train loss : {l.item():.5f} param_opt {Trained_End2EndWFS.WFSmodule.WFS.param.tolist()}")
            loss_tracker.append(l.item())
            param_tracker.append(Trained_End2EndWFS.WFSmodule.WFS.param.tolist())
        
        final_train_loss = l +final_train_loss
        
    return loss_tracker, np.array(param_tracker)




if __name__ == "__main__":
    
    device = 'cuda' # set to "cpu" if Cuda is not available
    
    paramfile = 'params_exp.py'  # file of experimental parameters

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
                           device)
    
    
    # Setting the loss function
    loss = torch.nn.L1Loss()
    
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
    optimizer_o = torch.optim.AdamW([Trained_End2EndWFS.WFSmodule.WFS.param],lro)

   
    optimizer_n = torch.optim.AdamW(Trained_End2EndWFS.PhaseEstimator.parameters(),lrn)
 
   
    # Training part for parameters optimization
    print("Initialized parameters",Trained_End2EndWFS.WFSmodule.param)
    
    
    checkpoint_path = 'ZernikeTestNoise6.pth'
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
    train_loss, train_parameters = train(Trained_End2EndWFS,dataset,loss,TrainRunNb,optimizer_o,optimizer_n,device)
    b = time.time() - a 
    
    # Testing part
    test_loss = test(Trained_End2EndWFS,dataset,loss,TestRunNb,device)
    
    # Save network and parameters
    
  
    torch.save(Trained_End2EndWFS.state_dict(), 'example'+'.pth' )
    
    # To load a network and have a look on the parameters :
    
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
        