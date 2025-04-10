#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:45:31 2025

@author: ptrouve
"""

from TorchPropagator import WFS
from PhaseEstimators import LinearEstimator, SimpleNet, ViT_PyTorch
from MaskGeneration import MaskGenerator
import torch.nn as nn
import torch
from mmengine import Config
import os

from line_profiler import profile


    
     
class WFSmodule (nn.Module) :
    
    def __init__(self, ParamsDict,device) :
        
        super().__init__()
        

        self.param = nn.Parameter(torch.tensor(ParamsDict['InitParam']).to(device))
        self.param_name = "toy example parameter vector"
        
        self.WFS = WFS(ParamsDict['Nres'],ParamsDict['sampling'],ParamsDict['D'],ParamsDict['Nphotons'], ParamsDict['RON'],ParamsDict['useNoise'],self.param,ParamsDict['MaskType'],device)
        
        
    def forward(self, phase):
        
        
        return self.WFS.Propagator(phase)     


    
class End2EndWFS (nn.Module):
    
    
    def __init__(self, ParamsDict,device, phaseEstimator = "SimpleNet"):
        
        super().__init__()
     
        self.WFSmodule = WFSmodule(ParamsDict,device)
        self.phaseEstimator = phaseEstimator
        # Choice of phase estimator (Linear or learned optimized)
        Nzernike = ParamsDict['Nzernike']
        self.maskType = ParamsDict['MaskType']
        # change here the processing for phase estimation
        
        if phaseEstimator == "Linear":
            self.PhaseEstimator = LinearEstimator(self.WFSmodule.WFS)
        elif phaseEstimator == "SimpleNet":
            self.PhaseEstimator = SimpleNet(Nzernike).to(device)
        elif phaseEstimator == "Transformer":
            self.PhaseEstimator = ViT_PyTorch(embed_dim = 256,
                                              img_size = 100,
                                              patch_size = 10,
                                              dropout = 0.2,
                                              num_heads = 4,
                                              num_encoders = 4,
                                              expansion = 2,
                                              Nzernike = Nzernike).to(device)
        
        
        if self.maskType != "Pyramid" or self.maskType != "Zernike":
            self.N = ParamsDict['Nres'] * ParamsDict['sampling']  # Resolution
            
            # Generate a grid of frequency coordinates
            u = torch.linspace(-1, 1, self.N, device = device)  # Normalized frequency range
            U, V = torch.meshgrid(u, u, indexing="xy")  # Create the full grid
            
            self.circ_mask = (torch.sqrt(U ** 2 + V ** 2) < 1).flatten()
            
            # Flatten and stack into (N^2, 2) shape
            self.uv_coords = torch.stack([U.flatten(), V.flatten()], dim=1).to(device)
        
        
        if self.maskType == "FreePhase":
            self.phaseMaskGenerator = MaskGenerator(isPhaseMask = True).to(device)
            self.phaseMask = self.phaseMaskGenerator(self.uv_coords).view(self.N, self.N)
       
        
        elif self.maskType == "FreeTransmision":
            self.transmisionMaskGenerator = MaskGenerator(isPhaseMask = False).to(device)
            self.transmisionMask = self.transmisionMaskGenerator(self.uv_coords).view(self.N, self.N)
            
        elif self.maskType == "FreePhaseTransmision":
            self.phaseMaskGenerator = MaskGenerator(isPhaseMask = True).to(device)
            self.phaseMask = self.phaseMaskGenerator(self.uv_coords).view(self.N, self.N)
            self.transmisionMaskGenerator = MaskGenerator(isPhaseMask = False).to(device)
            self.transmisionMask = self.transmisionMaskGenerator(self.uv_coords).view(self.N, self.N)

        
    def forward(self, x):
        # input : x (tensor) : input phase
        # output : estimated phase
        
        # Reload the ZernikeMask according to current parameter
         if self.maskType == "Pyramid":
             
             self.WFSmodule.WFS.BuildPyramidMask()
             
         elif self.maskType == "Zernike":
             self.WFSmodule.WFS.BuildZernikeMask()
         
            
         elif self.maskType == "FreePhase":
             self.phaseMask = self.phaseMaskGenerator(self.uv_coords)
             self.phaseMask = self.remove_tip_tilt(self.phaseMask).view(self.N, self.N)
             self.WFSmodule.WFS.SetMask(phaseMask = self.phaseMask)
             
         elif self.maskType == "FreeTransmision":  
             self.transmisionMask = self.transmisionMaskGenerator(self.uv_coords).view(self.N, self.N)
             self.WFSmodule.WFS.SetMask(transmisionMask = self.transmisionMask)
             
         elif self.maskType == "FreePhaseTransmision":
             self.phaseMask = self.phaseMaskGenerator(self.uv_coords)
             self.phaseMask = self.remove_tip_tilt(self.phaseMask).view(self.N, self.N)
             self.transmisionMask = self.transmisionMaskGenerator(self.uv_coords).view(self.N, self.N)
             self.WFSmodule.WFS.SetMask(phaseMask = self.phaseMask, transmisionMask = self.transmisionMask)
         
         # Compute the image from the phase
        
         self.Image = self.WFSmodule(x)       
         
         # Estimate the phase 
         
         EstimatedPhase = self.PhaseEstimator(self.Image)
         
         return EstimatedPhase
     

    def remove_tip_tilt(self, mask):
        """ Removes only the tip & tilt (ignores bias/constant offset). """
    
        # Solve least squares
        coeffs = torch.linalg.lstsq(self.uv_coords[self.circ_mask], mask[self.circ_mask]).solution  # coeffs.shape = [2, 1]
    
        # Compute the tilt plane using only (u, v) terms
        tilt_plane = self.uv_coords @ coeffs  # Result: [N, 1]
    
        # Remove tip & tilt
        mask_corrected = mask - tilt_plane
        return mask_corrected   
     
        
class AOLoop:
    def __init__(self, End2EndWFS, z_FullRes, gain, phaseTemplate, outputTemplate, photons, ron, start_after_iteration = 0, modulation = 0):
        
        self.End2EndWFS = End2EndWFS
        self.z_FullRes = z_FullRes
        
        if self.End2EndWFS.WFSmodule.WFS.maskType == "Pyramid" and self.End2EndWFS.phaseEstimator == "Linear":
            self.End2EndWFS.WFSmodule.WFS.param = [0.78, 0.78]
            self.End2EndWFS.WFSmodule.WFS.modulation = modulation
            self.End2EndWFS.WFSmodule.WFS.BuildPyramidMask()
            Nres = self.End2EndWFS.WFSmodule.WFS.Nres
            self.End2EndWFS.WFSmodule.WFS.BuildReconstructionMatrix(self.z_FullRes.view(-1,Nres,Nres))        
            self.End2EndWFS.WFSmodule.WFS.BuildReferenceIntensity()
        
        
        self.gain = gain
        self.z_estimated = torch.zeros_like(outputTemplate)  # Start with zero correction        
        self.z_reconstructed = torch.zeros_like(phaseTemplate)
        self.residual_phase = torch.zeros_like(phaseTemplate)  # Start with the original phase    
        self.pupil = self.End2EndWFS.WFSmodule.WFS.pupil
        self.residual_variance = torch.var(phaseTemplate[:, self.pupil.bool()], dim=-1).unsqueeze(-1)
        self.start_after_iteration = start_after_iteration
        
        self.End2EndWFS(phaseTemplate)
        self.images = self.End2EndWFS.WFSmodule(phaseTemplate)
        self.End2EndWFS.WFSmodule.WFS.SetPhotonsAndRON(photons, ron)
        self.iteration = 0

        
        
            
        
        
    def step(self, phase):
        self.iteration += 1
        self.residual_phase = phase - self.z_reconstructed
        self.residual_variance = torch.cat((self.residual_variance, torch.var(self.residual_phase[:, self.pupil.bool()], dim = -1).unsqueeze(-1)), dim = 1)
        self.images = self.End2EndWFS.WFSmodule(self.residual_phase)
        
        if self.iteration > self.start_after_iteration:
            z_output = self.End2EndWFS.PhaseEstimator(self.images)
            self.z_estimated = 1 * self.z_estimated + self.gain * z_output
            self.z_reconstructed = torch.matmul(self.z_estimated, self.z_FullRes).view_as(self.z_reconstructed)
            


    def ResetAOLoop(self):
        self.iteration = 0
        self.z_reconstructed = self.z_reconstructed * 0
        self.z_estimated = self.z_estimated * 0

        
    

class CheckpointManager:
    def __init__(self, model, WFSParams, TrainParams, checkpoint_path, optimizer_o = None, optimizer_n = None):
        self.model = model
        self.optimizer_o = optimizer_o
        self.optimizer_n = optimizer_n
        self.WFSParams = WFSParams
        self.TrainParams = TrainParams
        self.checkpoint_path = checkpoint_path

    def load(self):
        """Load checkpoint from the given path"""
        
        if not os.path.exists(self.checkpoint_path):
            print("Starting from scratch")
            return
        
        self.load_network(self.checkpoint_path)
        
        if self.WFSParams['MaskType'] == "FreePhase":
            self.load_free_phaseMask(self.checkpoint_path)

        elif self.WFSParams['MaskType'] == "FreeTransmision":
            self.load_free_transmisionMask(self.checkpoint_path)
        
        elif self.WFSParams['MaskType'] == "FreePhaseTransmision":
            self.load_free_phaseMask(self.checkpoint_path)
            self.load_free_transmisionMask(self.checkpoint_path)
            
        else:
            self.load_parametric_mask(self.checkpoint_path)



    def load_network(self, network_path = None, should_load_optimizer = True):
        
        if network_path is None:
            network_path = self.checkpoint_path
            
        if not os.path.exists(network_path):
            print(f'The path {network_path} does not exist')
            return
        
        checkpoint = torch.load(network_path)
        self.model.PhaseEstimator.load_state_dict(checkpoint['PhaseEstimator_state_dict'])
        if should_load_optimizer:
            self.optimizer_n.load_state_dict(checkpoint['optimizer_n_state_dict'])
            for param_group in self.optimizer_n.param_groups:
                param_group['lr'] = self.TrainParams["lrn"]

    def load_free_phaseMask(self, mask_path = None, should_load_optimizer = True):
        if mask_path is None:
            mask_path = self.checkpoint_path
        
        if not os.path.exists(mask_path):
            print(f'The path {mask_path} does not exist')
            return
        
        checkpoint = torch.load(mask_path)
        self.model.phaseMaskGenerator.load_state_dict(checkpoint['Phase_Mask_state_dict'])
        if should_load_optimizer:
            self.optimizer_o.load_state_dict(checkpoint['optimizer_o_state_dict'])
            for param_group in self.optimizer_o.param_groups:
                param_group['lr'] = self.TrainParams["lro"]
            
            
    def load_free_transmisionMask(self, mask_path = None, should_load_optimizer = True):
        if mask_path is None:
            mask_path = self.checkpoint_path
        
        if not os.path.exists(mask_path):
            print(f'The path {mask_path} does not exist')
            return
        
        checkpoint = torch.load(mask_path)
        self.model.transmisionMaskGenerator.load_state_dict(checkpoint['Transmision_Mask_state_dict'])
        if should_load_optimizer:
            self.optimizer_o.load_state_dict(checkpoint['optimizer_o_state_dict'])
            for param_group in self.optimizer_o.param_groups:
                param_group['lr'] = self.TrainParams["lro"]
            
            
               
            
    def load_parametric_mask(self, mask_path = None, should_load_optimizer = True):
        if mask_path is None:
            mask_path = self.checkpoint_path
            
        if not os.path.exists(mask_path):
            print(f'The path {mask_path} does not exist')
            return
        
        checkpoint = torch.load(mask_path)
        self.model.WFSmodule.load_state_dict(checkpoint['Mask_state_dict'])
        if should_load_optimizer:
            self.optimizer_o.load_state_dict(checkpoint['optimizer_o_state_dict'])
            for param_group in self.optimizer_o.param_groups:
                param_group['lr'] = self.TrainParams["lro"]
    
    def load_model(self, model_path = None, should_load_optimizer = True):
        if model_path is None:
            model_path = self.checkpoint_path
            
        if not os.path.exists(model_path):
            print(f'The path {model_path} does not exist')
            return
    
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not should_load_optimizer:
            return
        self.optimizer_o.load_state_dict(checkpoint['optimizer_o_state_dict'])
        self.optimizer_n.load_state_dict(checkpoint['optimizer_n_state_dict'])
        for param_group in self.optimizer_n.param_groups:
            param_group['lr'] = self.TrainParams["lrn"]
        for param_group in self.optimizer_o.param_groups:
            param_group['lr'] = self.TrainParams["lro"]

    def save(self, save_path = None):
        
        if save_path is None:
            save_path = self.checkpoint_path
        
        """Save checkpoint to the given path"""
        dict_to_save = {}
        dict_to_save['PhaseEstimator_state_dict'] = self.model.PhaseEstimator.state_dict()
        dict_to_save['optimizer_o_state_dict'] = self.optimizer_o.state_dict()
        dict_to_save['optimizer_n_state_dict'] = self.optimizer_n.state_dict()
        
        
        if self.WFSParams['MaskType'] == "FreePhase":
            dict_to_save['Phase_Mask_state_dict'] = self.model.phaseMaskGenerator.state_dict()

        elif self.WFSParams['MaskType'] == "FreeTransmision":
            dict_to_save['Transmision_Mask_state_dict'] = self.model.transmisionMaskGenerator.state_dict()

        
        elif self.WFSParams['MaskType'] == "FreePhaseTransmision":
            dict_to_save['Phase_Mask_state_dict'] = self.model.phaseMaskGenerator.state_dict()
            dict_to_save['Transmision_Mask_state_dict'] = self.model.transmisionMaskGenerator.state_dict()

            
        else:
            dict_to_save['Mask_state_dict'] = self.model.WFSmodule.state_dict()
        

        torch.save(dict_to_save, save_path)

     


         
if __name__ == "__main__":
    
    device = 'cuda'
    
    paramfile = 'params_exp.py'

    WFSParams = Config.fromfile(paramfile)['WFSParams']
 
    
    MyEnd2EndWFS = End2EndWFS(WFSParams,device)
    
    
    # for p in MyEnd2EndWFS.PhaseEstimator.parameters():
    #     if p.requires_grad:
    #         print(p.name, p.data)