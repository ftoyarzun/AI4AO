# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:37:06 2025

@author: franc
"""

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import time
import os

import imageio.v2 as imageio


class MaskGenerator(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),  
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  
        )

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.5)  # Normal distribution
            nn.init.zeros_(module.bias)  # Set bias to zero

    def forward(self, uv_coords):
        return self.net(uv_coords)

def train (maskGenerator, uv_coords, mask, loss, TrainRunNb, optimizer, device = 'cuda'):
    
    final_train_loss = 0
    
    fig,ax = plt.subplots(figsize = (10,10))
    
    img = ax.imshow(maskGenerator(uv_coords).view(N,N).cpu().detach().numpy())
    fig.colorbar(img)
    plt.show()
    
    writer = imageio.get_writer(gif_path, mode="i", fps = 10)
    
    maskGenerator.train()

    for u in range(0,TrainRunNb) :
        

        optimizer.zero_grad()

        output = maskGenerator(uv_coords).view(N, N)  
        
   
        # take a batch of images to estimage a batch of zernike parameters
     
        l = loss(output,mask.to(output.dtype))
   
          
        l. backward()
        
        #Comment one of these line to stop optimization on either part
        
        optimizer.step()

        
        # with torch.no_grad():  # Clip the parameter values after optimization step
        
        #     for param in End2EndWFS.WFSmodule.WFS.param:
        #         param.clamp_(0.001, 1000)
        
        

        # parameters values should change during the loop
        #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  param_proc  : {:.7f}\n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item(),Trained_End2EndWFS.PhaseEstimator.param[0,0].item()), end="")
        if u % 100 == 0:
            #print(" Run n°  {}, train loss : {:.7f} param_opt 1 :   {:.7f} param_opt 2 :   {:.7f}  \n".format(u, l,Trained_End2EndWFS.WFSmodule.WFS.param[0].item(),Trained_End2EndWFS.WFSmodule.WFS.param[1].item()), end="")
            print(f" Run n°  {u}, train loss : {l.item():.5f}")
            img.set_data(maskGenerator(uv_coords).view(N,N).cpu().detach().numpy())
            img.set_clim(vmin=np.min(img.get_array()), vmax=np.max(img.get_array()))
            
            fig.canvas.draw()
            image = np.array(fig.canvas.buffer_rgba())
            writer.append_data(image)
            
            plt.pause(0.1)
            #loss_tracker.append(l.item())
            #param_tracker.append(Trained_End2EndWFS.WFSmodule.WFS.param.tolist())
        
        final_train_loss = l +final_train_loss
        writer.close()
    return 


if __name__ == "__main__":
    
    device = 'cuda' # set to "cpu" if Cuda is not available
    
    paramfile = 'params_exp.py'  # file of experimental parameters

    gif_path = 'test_mask_animation.gif'
    
    mask_path = 'Pyramid_mask.pth'
    # Config extraction
   
    N = 200
    
    # Setting the loss function
    loss = torch.nn.MSELoss()
    
    # Initialisation of the system 
    maskGenerator = MaskGenerator().to(device)
    
    u = torch.linspace(-1, 1, N)  # Normalized frequency range
    v = torch.linspace(-1, 1, N)
    U, V = torch.meshgrid(u, v, indexing="xy")  # Create the full grid
    
    pyr_mask = (np.pi / 4 * (torch.abs(U) + torch.abs(V)) * N/2).to(device)
    zernike_mask = np.pi/2 * (torch.sqrt(U**2 + V**2) < 10/N).to(device)
    random_mask = torch.randn_like(zernike_mask)
    
    uv_coords = torch.stack([U.flatten(), V.flatten()], dim=1).to(device)

    # Flatten and stack into (N^2, 2) shape
    
   
    
    # Optimization parameters (learning rate lr and nb of runs)
   
   
    optimizer = torch.optim.Adam(maskGenerator.parameters(),0.001)

    
    a = time.time()
    train_loss = train(maskGenerator, uv_coords, pyr_mask,loss,5000,optimizer,device)
    b = time.time() - a 
    
    torch.save({
        'Mask_state_dict': maskGenerator.state_dict(),
        'optimizer_o_state_dict': optimizer.state_dict()
        }, mask_path)
    
