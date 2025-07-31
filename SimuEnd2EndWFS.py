#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:45:31 2025

@author: ptrouve
"""

from TorchPropagator import WFS
from PhaseEstimators import LinearEstimator, SimpleNet, ViT_PyTorch, DataFusion, Papyrus, VGGNet, PapyrusPhase, UNetWithMLP
from MaskGeneration import MaskManager
import torch.nn as nn
import torch
from mmengine import Config
import os
from Constants import mask_types_list, param_needed_mask_list
import torch.nn.functional as F

from line_profiler import profile


    
     
# class WFSmodule (nn.Module) :
    
#     def __init__(self, ParamsDict,device) :
        
#         super().__init__()
#         self.param = nn.Parameter(torch.tensor(ParamsDict['InitParam']).to(device))
#         self.param_name = "toy example parameter vector"       
#         self.WFS = WFS(ParamsDict, self.param, device)
#     def forward(self, phase):

#         return self.WFS.Propagator(phase)     
        
    
class End2EndWFS(nn.Module):
    def __init__(self, wfsParams, atmosParams, device):
        super().__init__()
        self.device = device
        self.maskType = wfsParams['MaskType']
        self.Nzernike = wfsParams['Nzernike']
        self.N = wfsParams['Nres'] * wfsParams['sampling']
        self.Nres = wfsParams['Nres']
        self.WFS = WFS(wfsParams, device)
        self.ReconstructionType = wfsParams['Reconstruction']
        #self.phaseEstimator = ParamsDict['Reconstruction']

        # Phase Estimator Selection
        self.PhaseEstimator = self._build_phase_estimator(wfsParams, atmosParams, device)

        # Initialize Mask Manager
        self.maskManager = MaskManager(wfsParams, device, self.WFS)
        self.maskManager.update_masks()
        

    def _build_phase_estimator(self, wfsParams, atmosParams, device):
        if wfsParams['Reconstruction'] == "Linear":
            return LinearEstimator(self.WFS)
        elif wfsParams['Reconstruction'] == "SimpleNet":
            return SimpleNet(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams['Reconstruction'] == "DataFusion":
            return DataFusion(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams['Reconstruction'] == "Papyrus":
            return Papyrus(wfsParams, atmosParams, device).to(self.device)
        elif wfsParams['Reconstruction'] == "PapyrusPhase":
            return PapyrusPhase(self.WFS.pupil, device).to(self.device)
        elif wfsParams['Reconstruction'] == "VGGNet":
            return VGGNet(wfsParams).to(self.device)
        elif wfsParams['Reconstruction'] == "UNet":
            return UNetWithMLP(1, mlp_output_size = wfsParams["Nzernike"]).to(self.device)
        elif wfsParams['Reconstruction'] == "Transformer":
            return ViT_PyTorch(
                embed_dim=256,
                img_size=80,
                patch_size=8,
                dropout=0.2,
                num_heads=4,
                num_encoders=4,
                expansion=2,
                Nzernike=self.Nzernike,
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown phase estimator: {wfsParams['Reconstruction']}")


    @profile
    def forward(self, x):
        linearEstimation = 0
        self.maskManager.update_masks()
        self.Image = self.WFS.Propagator(x)
        self.focalPlaneImage = self.WFS.ufocal
        input_to_network = self.Image
        
        if self.ReconstructionType == "DataFusion":
            input_to_network = torch.cat((input_to_network.unsqueeze(1), self.focalPlaneImage.unsqueeze(1)), dim = 1)
        if self.ReconstructionType == "Papyrus":
            input_to_network = self.bin_image(self.MergePupils(),2)
        if self.ReconstructionType == "Transformer":
            input_to_network = self.bin_image(self.MergePupils(),2)
        if self.ReconstructionType == "PapyrusPhase":
            input_to_network = self.bin_image(self.MergePupils(),2)
        if self.ReconstructionType == "UNet":
            input_to_network = self.bin_image(self.MergePupils(),2)    
    
        EstimatedPhase = self.PhaseEstimator(input_to_network)
        return EstimatedPhase
         


    def GetPupils(self, images = None):
        
        if images is None:
            images = self.Image
        
        images = images - self.WFS.reference_intensity.detach()
        
        Ncrop = self.Nres + 2

        centers = self.maskManager.GetPupilCenter()

        out = torch.zeros((images.shape[0], 4, Ncrop, Ncrop), device = self.device)
        
        for i, center in enumerate(centers):
            out[:, i] = images[..., center[0] - Ncrop//2: center[0] + Ncrop//2, center[1] - Ncrop//2: center[1] + Ncrop//2]

        return out
    
    def MergePupils(self, pupils = None):
        
        if pupils is None:
            pupils = self.GetPupils()
        
        Ncrop = pupils.shape[-1]
        
        out = torch.zeros((pupils.shape[0], Ncrop * 2, Ncrop * 2), device = self.device)
        
        
        
        out[..., Ncrop:, Ncrop:] = pupils[:, 0]
        out[..., :Ncrop, Ncrop:] = pupils[:, 1]
        out[..., :Ncrop, :Ncrop] = pupils[:, 2]
        out[..., Ncrop:, :Ncrop] = pupils[:, 3]
        
        return out
    
    def GetPhaseVariance(self, phase):
        return torch.var(phase[..., self.WFS.pupil.bool()], dim = -1, keepdim=True).unsqueeze(-1)
    
    
    def bin_image(self, image: torch.Tensor, bin_size: int) -> torch.Tensor:
        """
        Bins a 2D image (or batch of images) by summing over bin_size x bin_size regions.
    
        Args:
            image (torch.Tensor): shape (B, C, H, W)
            bin_size (int): binning factor
    
        Returns:
            torch.Tensor: binned image of shape (B, C, H//bin_size, W//bin_size)
        """
        B, H, W = image.shape
        image = image.unsqueeze(1)  # Add channel dimension: (B, 1, H, W)
    
        # Create uniform binning kernel
        kernel = torch.ones((1, 1, bin_size, bin_size), device=image.device)
    
        # Apply convolution with stride = bin_size
        binned = F.conv2d(image, kernel, stride=bin_size)

    
        return binned.squeeze(1)  # Remove channel dimension -> (B, H//bin, W//bin)
        
class AOLoop:
    def __init__(self, ParamsDict, End2EndWFS, z_FullRes, gain, phaseTemplate, outputTemplate, photons, ron, start_after_iteration = 0, modulation = 0):
        
        self.End2EndWFS = End2EndWFS
        self.z_FullRes = z_FullRes
        
        if ParamsDict["Reconstruction"] == "Linear":
            self.End2EndWFS.WFS.param = [torch.pi/2, torch.pi/2]
            self.End2EndWFS.WFS.modulation = modulation
            self.End2EndWFS.WFS.BuildPyramidMask()
            Nres = self.End2EndWFS.WFS.Nres
            self.End2EndWFS.WFS.BuildReconstructionMatrix(self.z_FullRes.view(-1,Nres,Nres))        
            self.End2EndWFS.WFS.BuildReferenceIntensity()
        
        
        self.gain = gain
        self.z_estimated = torch.zeros_like(outputTemplate)  # Start with zero correction        
        self.z_reconstructed = torch.zeros_like(phaseTemplate)
        self.residual_phase = torch.zeros_like(phaseTemplate)  # Start with the original phase    
        self.pupil = self.End2EndWFS.WFS.pupil
        self.residual_variance = torch.var(phaseTemplate[:, self.pupil.bool()], dim=-1).unsqueeze(-1)
        self.start_after_iteration = start_after_iteration
        
        self.End2EndWFS(phaseTemplate)
        self.images = self.End2EndWFS.Image
        self.End2EndWFS.WFS.SetPhotonsAndRON(photons, ron)
        self.iteration = 0

        
        
            
        
        
    def step(self, phase):
        self.iteration += 1
        self.residual_phase = phase - self.z_reconstructed
        self.residual_variance = torch.cat((self.residual_variance, torch.var(self.residual_phase[:, self.pupil.bool()], dim = -1).unsqueeze(-1)), dim = 1)
        z_output = self.End2EndWFS(self.residual_phase)
        self.images = self.End2EndWFS.Image
        
        if self.iteration > self.start_after_iteration:
            self.z_estimated = 0.999 * self.z_estimated + self.gain * z_output
            self.z_reconstructed = torch.matmul(self.z_estimated, self.z_FullRes).view_as(self.z_reconstructed)
            


    def ResetAOLoop(self):
        self.iteration = 0
        self.z_reconstructed = self.z_reconstructed * 0
        self.z_estimated = self.z_estimated * 0

        
    

class CheckpointManager:
    def __init__(self, model, WFSParams, TrainParams, checkpoint_path, optimizer_o = None, optimizer_n = None):
        self.model = model
        self.maskManager = model.maskManager
        self.optimizer_o = optimizer_o
        self.optimizer_n = optimizer_n
        self.WFSParams = WFSParams
        self.TrainParams = TrainParams
        self.checkpoint_path = checkpoint_path

    def load(self, should_load_optimizer = True):
        """Load checkpoint from the given path"""
        
        if not os.path.exists(self.checkpoint_path):
            print("Starting from scratch")
            return
        
        self.load_network(self.checkpoint_path, should_load_optimizer)
        
        if self.WFSParams['MaskType'] in ["FreePhase", "FreePhaseTransmision", "ModalMask"]:
            self.load_free_phaseMask(self.checkpoint_path, should_load_optimizer)

        if self.WFSParams['MaskType'] in ["FreeTransmision", "FreePhaseTransmision"]:
            self.load_free_transmisionMask(self.checkpoint_path, should_load_optimizer)
        
        if self.WFSParams['MaskType'] in param_needed_mask_list:
            self.load_parametric_mask(self.checkpoint_path, should_load_optimizer)

        if self.WFSParams['MaskType'] not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.WFSParams['MaskType']}")


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
        self.maskManager.phaseMaskGenerator.load_state_dict(checkpoint['Phase_Mask_state_dict'])
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
        self.maskManager.transmisionMaskGenerator.load_state_dict(checkpoint['Transmision_Mask_state_dict'])
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
        self.maskManager.load_state_dict(checkpoint['Mask_state_dict'])
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
        
        
        if self.WFSParams['MaskType'] in ["FreePhase", "FreePhaseTransmision"]:
            dict_to_save['Phase_Mask_state_dict'] = self.maskManager.phaseMaskGenerator.state_dict()

        if self.WFSParams['MaskType'] in ["FreeTransmision", "FreePhaseTransmision"]:
            dict_to_save['Transmision_Mask_state_dict'] = self.maskManager.transmisionMaskGenerator.state_dict()

        if self.WFSParams['MaskType'] == "ModalMask":
            dict_to_save['Phase_Mask_state_dict'] = self.maskManager.phaseMaskGenerator.state_dict()
    
        if self.WFSParams['MaskType'] in param_needed_mask_list:
            dict_to_save['Mask_state_dict'] = self.maskManager.state_dict()
        
        if self.WFSParams['MaskType'] not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.WFSParams['MaskType']}")

        torch.save(dict_to_save, save_path)

     


         
if __name__ == "__main__":
    
    device = 'cuda'
    
    paramfile = 'params_exp.py'

    WFSParams = Config.fromfile(paramfile)['WFSParams']
 
    
    MyEnd2EndWFS = End2EndWFS(WFSParams,device)
    
    
    # for p in MyEnd2EndWFS.PhaseEstimator.parameters():
    #     if p.requires_grad:
    #         print(p.name, p.data)