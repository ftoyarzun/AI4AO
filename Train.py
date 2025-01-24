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


def test (End2EndWFS, dataset, loss, TestRunNb, device = 'cuda'):
    
    L = 0
    phaseGT,zernike,_,_=dataset[0]
      
       
    output = End2EndWFS(phaseGT[0,:,:]) 
    
    fig, axes = plt.subplots(1, 1, figsize=(13, 5))
    line1, = axes.plot(zernike[0,:].cpu().data.numpy())
    line2, = axes.plot(output.cpu().data.numpy())
    

    
    for u in range(0,TestRunNb) :
        
        End2EndWFS.train()
        
     
        phaseGT,zernike,_,_=dataset[0]
      
       
        output = End2EndWFS(phaseGT[0,:,:])  
        
   
        #here it should take a batch of ground truth phase and return estimated zernike coeffs - here only a single image
        
    
        l = loss(output,zernike[0,:])
  

        line1.set_ydata(zernike[0,:].cpu().data.numpy())
        line2.set_ydata(output.cpu().data.numpy())
        plt.xlabel('Zernike mode index')
        plt.ylabel('Zernike mode Amplitude')
        plt.pause(0.1)
        plt.show()

        
        print(" Test Run n°  {}, train loss : {:.4f}  \n".format(u, l), end="")
        
   
    return l


def train (End2EndWFS, dataset, loss, TrainRunNb, optimizer_o,optimizer_n, device = 'cuda'):
    
    L = 0
    
    for u in range(0,TrainRunNb) :
        
        
        End2EndWFS.train()
        
        #optimizer_n.zero_grad()
        optimizer_o.zero_grad()
        
        phaseGT,zernike,_,_=dataset[0]
      
       
        output = End2EndWFS(phaseGT[0,0,:,:])  
        
   
        #here it should take a batch of ground truth phase and return estimated zernike coeffs - here only a single image        
    
        l = loss(output,zernike[0,:])
   
          
        l. backward()
        
        #Comment one of these line to stop optimization on either part
        
        optimizer_o.step()
        #optimizer_n.step()

        
        print(" Run n°  {}, train loss : {:.7f}  \n".format(u, l),"param", Trained_End2EndWFS.WFS.param, end="")
        
   
    return l




if __name__ == "__main__":
    
    device = 'cpu' # set to "cpu" if Cuda is not available
    
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
    optimizer_o = torch.optim.SGD([Trained_End2EndWFS.WFS.param,Trained_End2EndWFS.WFS.param2],lro)
    optimizer_n = None
    #optimizer_n = torch.optim.SGD([Trained_End2EndWFS.PhaseEstimator.LearnedReconstructionMatrix],lrn)
 
   
    # Training part for parameters optimization  
    train_loss = train(Trained_End2EndWFS,dataset,loss,TrainRunNb,optimizer_o,optimizer_n,device)
       
    # Testing part
    test_loss = test(Trained_End2EndWFS,dataset,loss,TestRunNb,device)
     
        
        
        