#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:45:31 2025

@author: ptrouve
"""

from TorchPropagator import WFS, Zernike

import torch.nn as nn
import torch
from mmengine import Config
import os

from line_profiler import profile


class SimpleNet(nn.Module):
    def __init__(self, Nzernike):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride = 2),
            nn.GELU(),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride = 2),
            nn.GELU(), 

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), 
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride = 2),
            nn.GELU(), 

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), 
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=5, padding=2, stride = 2),
            nn.GELU(),   

            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),
            nn.Dropout(0.1)  
        )

        self.outputlayer = nn.Linear(256, Nzernike)

    def forward(self, x):
        x = x.unsqueeze(1).type(torch.float32)
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x


class PatchEmbedding(nn.Module):
  def __init__(self, embed_dim, img_size, patch_size, dropout, in_channels = 1):
      super().__init__()
      
      num_patches = (img_size // patch_size) ** 2
      
      self.patcher = nn.Sequential(
          # We use conv for doing the patching
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=embed_dim,
              # if kernel_size = stride -> no overlap
              kernel_size=patch_size,
              stride=patch_size
          ),
          # Linear projection of Flattened Patches. We keep the batch and the channels (b,c,h,w)
          nn.Flatten(2))
      self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
      self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
      self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
      # Create a copy of the cls token for each of the elements of the BATCH
      cls_token = self.cls_token.expand(x.shape[0], -1, -1)
      # Create the patches
      x = self.patcher(x).permute(0, 2, 1)
      # Unify the position with the patches
      x = torch.cat([cls_token, x], dim=1)
      # Patch + Position Embedding
      x = self.position_embeddings + x
      x = self.dropout(x)
      return x
  
    
class ViT_PyTorch(nn.Module):
    def __init__(self, embed_dim, img_size, patch_size, dropout, num_heads, num_encoders, expansion, Nzernike):
        super().__init__()
        
        self.inst_norm = nn.InstanceNorm2d(1)
        
        self.embeddings_block = PatchEmbedding(embed_dim, img_size, patch_size, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=int(embed_dim*expansion), activation="gelu", batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=Nzernike)
        )

    def forward(self, x):
        x = x.unsqueeze(1).type(torch.float32)
        x = self.inst_norm(x)
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # Apply MLP on the CLS token only
        return x



class OptimizedLinearEstimator (nn.Module) :
    " Learned Linear Estimator with a learned reconstruction matrix and ref intensity"
    "They are initalized using the propagator code from the starting point"
    
    def __init__(self,init=0,WFS=None,Nzernike=0) :
        
        super().__init__()
        
        # Initialization with the  reconstruction matrix at starting point
        if init == 1 :
            
            print("Initalization of the reconstruction matrix")
            [z, z_FullRes] = Zernike(WFS.pupil.cpu(), WFS.pupil_logical, WFS.Nres, Nzernike)     
            z_FullRes = z_FullRes
            WFS.BuildReconstructionMatrix(z_FullRes, WFS.mask)
            self.WFS = WFS
            self.param = nn.Parameter(WFS.reconstructionMatrix)
            self.param_name = "Reconstruction matrix as a parameter"
        # Reconstruction matrix initalized at 0
        else :
            number_of_pixels = WFS.Npix**2
            self.param = nn.Parameter(torch.zeros((Nzernike,number_of_pixels),dtype = torch.float64))
            
            
    def forward(self, image):
        
         ## (Learned) Matrix multiplication
         
        
         reduced_intensity= image
         
         EstimatedZernike = torch.matmul(reduced_intensity.flatten(start_dim = -2), self.param.T) 
         
         return EstimatedZernike
     
class LinearEstimator (nn.Module) :
    
    def __init__(self, WFS) :
        
        super().__init__()
        
        self.WFS = WFS       
        
    def forward(self, image):
        
         ## Build the reconstruction matrix for each forward (because it depends on the optimized parameters)
        # self.WFS.BuildReconstructionMatrix(self.z_FullRes, self.WFS.mask)
        
        return self.WFS.GetReconstructedPhase(image)
         
    
     
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
        
        
        if self.maskType == "Free":
            self.maskGenerator = MaskGenerator().to(device)
            
            self.N = ParamsDict['Nres'] * ParamsDict['sampling']  # Resolution
            
            # Generate a grid of frequency coordinates
            u = torch.linspace(-1, 1, self.N).to(device)  # Normalized frequency range
            U, V = torch.meshgrid(u, u, indexing="xy")  # Create the full grid
            
            self.circ_mask = (torch.sqrt(U ** 2 + V ** 2) < 1).flatten()
            
            # Flatten and stack into (N^2, 2) shape
            self.uv_coords = torch.stack([U.flatten(), V.flatten()], dim=1).to(device)
            self.mask = self.maskGenerator(self.uv_coords)
       
         
    def forward(self, x):
        # input : x (tensor) : input phase
        # output : estimated phase
        
        # Reload the ZernikeMask according to current parameter
         if self.maskType == "Pyramid":
             
             self.WFSmodule.WFS.BuildPyramidMask()
             
         elif self.maskType == "Zernike":
             self.WFSmodule.WFS.BuildZernikeMask()
         
            
         elif self.maskType == "Free":
             self.mask = self.maskGenerator(self.uv_coords)
             self.mask = self.remove_tip_tilt(self.mask).view(self.N, self.N)
             self.WFSmodule.WFS.SetMask(self.mask)
             
             
         # self.WFSmodule.WFS.BuildReferenceIntensity()
         
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
            self.End2EndWFS.WFSmodule.WFS.BuildReconstructionMatrix(self.z_FullRes.view(-1,Nres,Nres), self.End2EndWFS.WFSmodule.WFS.mask)        
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


class MaskGenerator(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),  # Input: (u, v)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Output: Mask value
        )

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Normal distribution
            nn.init.constant_(module.bias, 0.1)  # Set bias to zero

    def forward(self, uv_coords):
        return self.net(uv_coords)     
    

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
        
        if self.WFSParams['MaskType'] == "Free":
            self.load_free_mask(self.checkpoint_path)
            return
        
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

    def load_free_mask(self, mask_path = None, should_load_optimizer = True):
        if mask_path is None:
            mask_path = self.checkpoint_path
        
        if not os.path.exists(mask_path):
            print(f'The path {mask_path} does not exist')
            return
        
        checkpoint = torch.load(mask_path)
        self.model.maskGenerator.load_state_dict(checkpoint['Mask_state_dict'])
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
        
        
        if self.WFSParams['MaskType'] != "Free":
            dict_to_save['Mask_state_dict'] = self.model.WFSmodule.state_dict()
        else:
            dict_to_save['Mask_state_dict'] = self.model.maskGenerator.state_dict()
            
        torch.save(dict_to_save, save_path)

     


         
if __name__ == "__main__":
    
    device = 'cuda'
    
    paramfile = 'params_exp.py'

    WFSParams = Config.fromfile(paramfile)['WFSParams']
 
    
    MyEnd2EndWFS = End2EndWFS(WFSParams,device)
    
    
    # for p in MyEnd2EndWFS.PhaseEstimator.parameters():
    #     if p.requires_grad:
    #         print(p.name, p.data)