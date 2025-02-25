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

def test (End2EndWFS, dataset, loss, TestRunNb, device = 'cuda'):
    
  
    phaseGT,zernike,_,_=dataset[0]
      
    plt.close('all')   
    output = End2EndWFS(phaseGT[0,:,:]) 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
    img = axes[0].imshow(End2EndWFS.WFSmodule(phaseGT)[0,:,:].cpu().detach().numpy())
    line1, = axes[1].plot(zernike[0,:].cpu().data.numpy())
    line2, = axes[1].plot(output[0,:].cpu().data.numpy())
    
    axes[1].set_ylim(-2, 2)
    
    final_test_loss = 0
    
    End2EndWFS.eval()
    
    with torch.no_grad():
    
        for u in range(0,TestRunNb) :
 
            phaseGT,zernike,_,_=dataset[0]
            
           
          
           # take a batch of images to estimage a batch of zernike parameters
            output = End2EndWFS(phaseGT)  
            
            
    
            l = loss(output,zernike.to(output.dtype))
            
            final_test_loss = l +final_test_loss
            
            print(" Test Run n°  {}, test loss : {:.4f}  \n".format(u, l), end="")
            
            
            #plot of an example within the batch
            
            plt.subplot(1,2,1)
            img.set_data(End2EndWFS.WFSmodule(phaseGT)[0,:,:].cpu().detach().numpy())
        
            plt.subplot(1,2,2)
            line1.set_ydata(zernike[0,:].cpu().data.numpy())
            line2.set_ydata(output[0,:].cpu().data.numpy())
            plt.xlabel('Zernike mode index')
            plt.ylabel('Zernike mode Amplitude')
            plt.title(" MSE " + str(loss(output[0,:],zernike[0,:]).item()))
            
            plt.show()
            plt.pause(1)
            
            
   
    return final_test_loss /TestRunNb


def train (End2EndWFS, dataset, loss, TrainRunNb, optimizer_o,optimizer_n, device = 'cuda'):
    
    final_train_loss = 0
    
    End2EndWFS.train()
    
    for u in range(0,TrainRunNb) :
        
        
        
        
        optimizer_n.zero_grad()
        optimizer_o.zero_grad()
        
        phaseGT,zernike,_,_=dataset[0]
        
              
        output = End2EndWFS(phaseGT)  
        
   
        # take a batch of images to estimage a batch of zernike parameters
     
        l = loss(output,zernike.to(output.dtype))
   
          
        l. backward()
        
        #Comment one of these line to stop optimization on either part
        
        optimizer_o.step()
        optimizer_n.step()
        
        with torch.no_grad():  # Clip the parameter values after optimization step
        
            for param in End2EndWFS.WFSmodule.WFS.param:
                param.clamp_(0.001, 1000)
        
        

        # parameters values should change during the loop
        #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  param_proc  : {:.7f}\n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item(),Trained_End2EndWFS.PhaseEstimator.param[0,0].item()), end="")
        if u % 10 == 0:
            print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
        
        final_train_loss = l +final_train_loss
        
    return final_train_loss/TrainRunNb




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
    loss = torch.nn.MSELoss()
    
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
    optimizer_o = torch.optim.Adam([Trained_End2EndWFS.WFSmodule.WFS.param],lro)

   
    optimizer_n = torch.optim.Adam(Trained_End2EndWFS.PhaseEstimator.parameters(),lrn)
 
   
    # Training part for parameters optimization
    print("Initialized parameters",Trained_End2EndWFS.WFSmodule.param)
    
    
    checkpoint_path = 'checkpoint.pth'
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
    train_loss = train(Trained_End2EndWFS,dataset,loss,TrainRunNb,optimizer_o,optimizer_n,device)
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
    
    
        
        