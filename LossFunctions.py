# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:52:19 2025

@author: franc
"""

import torch
import torch.nn as nn


class Custom_Loss_Function(nn.Module):
    def __init__(self, epsilon = 1e-2, degree = 2, NZernike = 209, device = 'cuda'):
        super().__init__()
        self.epsilon = epsilon
        self.degree = degree
        self.NZernike = NZernike

        self.linspace = torch.sqrt((torch.linspace(1, NZernike, NZernike, device=device)))
        
    def forward(self, y_pred, y_true, r0):
        diff = (y_pred - y_true)[..., :self.NZernike] * self.linspace
        return torch.mean(torch.abs(diff) ** self.degree * r0 ** (5/3))



class Physics_loss(nn.Module):
    def __init__(self, z_fullRes, phase_template, degree = 2, device = 'cuda'):
        super().__init__()
        self.z_fullRes = z_fullRes
        self.phase_shape = phase_template.shape
        self.degree = degree
    
    def forward(self, WFSModule, I_WFS, y_pred, r0):
        
        I_pred = self.ComputeForwardImage(WFSModule, y_pred)
        return torch.mean(torch.abs(I_pred - I_WFS) ** self.degree * r0 ** (5/3)) * 1e6
        
    def ComputeForwardImage(self, WFSModule, y_pred):
        reconstructed_phase = torch.matmul(y_pred, self.z_fullRes).view(*self.phase_shape)
        
        return WFSModule(reconstructed_phase)


class ResidualPhaseLoss(nn.Module):
    def __init__(self, z_fullRes, pupil, device = 'cuda'):
        super().__init__()
        self.z_fullRes = z_fullRes
        self.pupil = pupil
    
    def forward(self, y_pred, phase, r0):
        reconstructed_phase = torch.matmul(y_pred, self.z_fullRes).view_as(phase)
        residual_phase = phase - reconstructed_phase
        return torch.mean(torch.var(residual_phase[..., self.pupil.bool()], dim = -1) * r0 ** (5/3))
        
     