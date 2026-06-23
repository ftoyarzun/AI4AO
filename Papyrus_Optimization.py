# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 16:11:26 2025

@author: foyarzun
"""

import torch


from SimuEnd2EndWFS import End2EndWFS, CheckpointManager

from MaskGeneration import MaskVisualizator
from mmengine import Config
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import loadmat



def trainMask (data, isLogical, Trained_End2EndWFS, optimizer):
    """
    Trains the MaskGenerator neural network to match a target mask pattern.
    Generates live plots and saves intermediate mask states into a GIF.

    Args:
        maskGenerator (nn.Module): Neural network to generate the mask.
        uv_coords (Tensor): 2D coordinates input to the mask generator.
        mask (Tensor): Target mask used for loss computation.
        loss (function): Loss function to optimize.
        TrainRunNb (int): Number of training iterations.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        device (str): Device for computation.

    Returns:
        None
    """
    final_train_loss = 0
    
    
    fig,ax = plt.subplots(figsize = (10,10))
    
    img = ax.imshow(data.cpu().detach().numpy())
    fig.colorbar(img)
    plt.show()
    
    Trained_End2EndWFS.maskManager.train()

    for u in range(0,TrainRunNb) :
        

        optimizer.zero_grad()
        
        Trained_End2EndWFS.maskManager.update_masks()

        Trained_End2EndWFS.WFS.BuildReferenceIntensity() 
        
        if isLogical:
            threshold = 1.5e-5
            digital_image = torch.sigmoid(1000000 * (Trained_End2EndWFS.WFS.reference_intensity - threshold))
        else:
            digital_image = Trained_End2EndWFS.WFS.reference_intensity
        
        

        l = loss(data,digital_image)
    
        l. backward()
        
        optimizer.step()
        
        if u % 10 == 0:
            #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {u}, train loss : {l.item():.5f}")
            img.set_data((data - digital_image).cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            
            plt.pause(0.1)
        
        # final_train_loss = l +final_train_loss
    return 


def TrainRooftop (data, Trained_End2EndWFS, optimizer):
    """
    Trains the MaskGenerator neural network to match a target mask pattern.
    Generates live plots and saves intermediate mask states into a GIF.

    Args:
        maskGenerator (nn.Module): Neural network to generate the mask.
        uv_coords (Tensor): 2D coordinates input to the mask generator.
        mask (Tensor): Target mask used for loss computation.
        loss (function): Loss function to optimize.
        TrainRunNb (int): Number of training iterations.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        device (str): Device for computation.

    Returns:
        None
    """
    final_train_loss = 0
    
    
    fig,ax = plt.subplots(figsize = (10,10))
    
    img = ax.imshow(data.cpu().detach().numpy())
    fig.colorbar(img)
    plt.show()
    
    Trained_End2EndWFS.maskManager.train()

    for u in range(0,TrainRunNb) :
        

        optimizer.zero_grad()
        
        Trained_End2EndWFS.maskManager.update_masks()

        Trained_End2EndWFS.WFS.BuildReferenceIntensity() 
        

        digital_image = Trained_End2EndWFS.WFS.reference_intensity
        
        pupils = Trained_End2EndWFS.GetPupils(digital_image.unsqueeze(0))
        pupils_ref = Trained_End2EndWFS.GetPupils(referenceFrame.unsqueeze(0))
        
        light_ratio_pupils = pupils.sum(dim = (-2,-1))
        light_ratio_pupils_ref = pupils_ref.sum(dim = (-2,-1))
        

        l = loss(light_ratio_pupils_ref,light_ratio_pupils)
    
        l. backward()
        
        optimizer.step()
        
        if u % 10 == 0:
            #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {u}, train loss : {l.item():.5f}")
            img.set_data((data - digital_image).cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            
            plt.pause(0.1)
        
        # final_train_loss = l +final_train_loss
    return 


def TrainDMFlat(referenceFrame, Trained_End2EndWFS, optimizer, dm_flat, loss, TrainRunNb=100):
    """
    Trains the deformable mirror (DM) flat map to match a reference wavefront sensor image.

    Args:
        referenceFrame (Tensor): Target image to match.
        Trained_End2EndWFS: AO system with WFS and Propagator.
        optimizer: Optimizer to update dm_flat.
        dm_flat (torch.nn.Parameter): Trainable DM map.
        loss_fn: Loss function (e.g., torch.nn.MSELoss()).
        TrainRunNb (int): Number of optimization iterations.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(referenceFrame.cpu().detach().numpy())
    fig.colorbar(img)
    plt.show()

    Trained_End2EndWFS.maskManager.train()

    for u in range(TrainRunNb):
        optimizer.zero_grad()

        Trained_End2EndWFS.maskManager.update_masks()
        
        frame = Trained_End2EndWFS.WFS.Propagator(dm_flat).squeeze()
 
        l = loss(referenceFrame, frame) * 1e10

        l.backward()
        
        optimizer.step()

        if u % 10 == 0:
            print(f"Run {u:03d} | Loss: {l.item():.5f}")
            img.set_data((referenceFrame - frame).cpu().detach().numpy())
            img.set_clim(vmin=img.get_array().min(), vmax=img.get_array().max())
            plt.pause(0.1)

    return



def fineAdjustmentMask (data, optimizer, index):
    """
    Trains the MaskGenerator neural network to match a target mask pattern.
    Generates live plots and saves intermediate mask states into a GIF.

    Args:
        maskGenerator (nn.Module): Neural network to generate the mask.
        uv_coords (Tensor): 2D coordinates input to the mask generator.
        mask (Tensor): Target mask used for loss computation.
        loss (function): Loss function to optimize.
        TrainRunNb (int): Number of training iterations.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        device (str): Device for computation.

    Returns:
        None
    """
    final_train_loss = 0
    
    
    # fig,ax = plt.subplots(figsize = (10,10))
    
    # img = ax.imshow(data.cpu().detach().numpy())
    # fig.colorbar(img)
    # plt.show()
    
    Trained_End2EndWFS.maskManager.train()

    for u in range(0,TrainRunNb) :
        

        optimizer.zero_grad()
        
        Trained_End2EndWFS.maskManager.update_masks()

        Trained_End2EndWFS.WFS.BuildReconstructionMatrix(papyrus_modal_dm[index]*amplitude, 30)# , (papyrus_zonal_dm  * dm_flat).sum(dim = 0))

        digital_image = Trained_End2EndWFS.WFS.iMat.view(-1,240, 240)
        
        

        l = loss(data,digital_image)*1e10
    
        l. backward() 
        
        optimizer.step()
        
        if u % 10 == 0:
            #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {u}, train loss : {l.item():.5f}")
            # img.set_data((data - digital_image).cpu().detach().numpy())
            # img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            
            plt.pause(0.1)
        
        # final_train_loss = l +final_train_loss
    return 



if __name__ == "__main__":
    
    device = 'cuda' # set to "cpu" if Cuda is not available
    
    paramfile = 'params_exp.py'  # file of experimental parameters

    gif_path = 'free_mask_animation_2ndStage3.gif'

    # Config extraction
    AtmosParams = Config.fromfile(paramfile)['AtmosParams']
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']
    TrainParams = Config.fromfile(paramfile)['TrainParams']
    
    
    data = torch.from_numpy(loadmat('../useful_pixels_20250604_0305.mat')['usefulPix']).to(device = device, dtype = torch.float32)
    iMat_Papyrus = torch.from_numpy(loadmat('../intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307')['matrix_inf']).to(device = device, dtype = torch.float32).view(240,240,-1).permute(2,0,1)
    referenceFrame = torch.from_numpy(loadmat('../referenceFrame_20250604_0306.mat')['referenceFrame']).to(device = device, dtype = torch.float32).view(240,240)
    referenceFrame /= referenceFrame.sum() 
    data = (referenceFrame.cpu() > 0.9e-5).to(device = device, dtype = torch.float32)
    
    #M2C = torch.from_numpy(np.load("../M2C.npy").astype(np.float32)).to(device=device)
    M2C = torch.from_numpy(loadmat('../M2C_KL_OOPAO_synthetic_IF')["M2C_KL"]).to(device = device, dtype = torch.float32)
    papyrus_dm = torch.from_numpy(np.load("../papyrus_dm.npy").astype(np.float32)).to(device=device) * 2e7

    papyrus_modal_dm = (papyrus_dm @ M2C).view(80,80,-1).permute(2,0,1)[:, 1:-1, 1:-1]
    papyrus_zonal_dm = papyrus_dm.view(80,80,-1).permute(2,0,1)[:, 1:-1, 1:-1]
    #papyrus_modal_dm = torch.flip(papyrus_modal_dm, dims=[1,2])


# %%


    # Dataset creationt.

    # Initialisation of the system 
    Trained_End2EndWFS = End2EndWFS(WFSParams, AtmosParams, device)
    
    Trained_End2EndWFS.WFS.modulation = 3

    
    TrainRunNb = 300
    
    loss = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam([Trained_End2EndWFS.maskManager.maskShifts],0.001)
    
    checkpoint_path = 'Papyrus.pth'
    checkpointManager = CheckpointManager(Trained_End2EndWFS, WFSParams, TrainParams, checkpoint_path, optimizer, optimizer)
    
    checkpointManager.load_parametric_mask(should_load_optimizer = False)
    
    
    Trained_End2EndWFS.maskManager.update_masks()
    Trained_End2EndWFS.WFS.BuildReferenceIntensity()
    
    #%%
    
    trainMask(data, True, Trained_End2EndWFS, optimizer)
    
# %%
    
    optimizer = torch.optim.Adam([Trained_End2EndWFS.maskManager.rooftop],0.01)
    TrainRooftop(referenceFrame, Trained_End2EndWFS, optimizer)
    
    
    
    #%%
    
    amplitude = torch.nn.Parameter(torch.tensor(1.))
    # dm_flat = torch.nn.Parameter(0.001 * torch.randn(241, 1, 1, device = device, dtype = torch.float32))
    index = range(40,90,5)
    optimizer = torch.optim.Adam([Trained_End2EndWFS.maskManager.maskShifts, amplitude, Trained_End2EndWFS.maskManager.rooftop],0.01)
    fineAdjustmentMask(iMat_Papyrus[index], optimizer, index)
    
    #%%

    dm_flat = torch.nn.Parameter(0.001 * torch.randn(241, 1, 1, device = device, dtype = torch.float32))
    optimizer = torch.optim.Adam([dm_flat, Trained_End2EndWFS.maskManager.maskShifts, Trained_End2EndWFS.maskManager.rooftop],0.01)
    TrainDMFlat(referenceFrame, Trained_End2EndWFS, optimizer, dm_flat, loss, TrainRunNb=300)

    