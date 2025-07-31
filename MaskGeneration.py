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

from Constants import mask_types_list

from TorchPropagator import ZernikeFullView

import imageio.v2 as imageio


class MaskManager(nn.Module):
    
    def __init__(self, ParamsDict, device, WFS):
        
        super().__init__()
        self.device = device
        self.maskType = ParamsDict['MaskType']
        self.sampling = ParamsDict['sampling']
        self.Nres = ParamsDict['Nres']
        self.N = int(self.Nres * self.sampling)
        self.WFS = WFS  # Reference to actual WFS object
        
        # Build uv grid for freeform mask types
        self._build_uv_grid()
        self._build_xy_grid()

        # Initialize mask generators
        self.phaseMask = None
        self.transmisionMask = None
        self._init_mask_generators(ParamsDict)

        # Placeholder for masks
        

    def _build_uv_grid(self):
        u = torch.linspace(-1, 1, self.N, device=self.device)
        U, V = torch.meshgrid(u, u, indexing="xy")
        self.UV = torch.stack([U.flatten(), V.flatten()], dim=1)  # (N², 2)
        self.circ_mask = (torch.sqrt(U ** 2 + V ** 2) < 1).flatten()  # (N²,)
        
    
    
    def _build_xy_grid(self):
        x_mask = torch.linspace(-self.N/2, self.N/2-1, self.N, dtype=torch.float32, device = self.device)
        [self.x_mask,self.y_mask] = torch.meshgrid(x_mask,x_mask)

        self.rho_mask = torch.sqrt(self.x_mask ** 2 + self.y_mask ** 2)
        self.abs_x_mask = torch.abs(self.x_mask)
        self.abs_y_mask = torch.abs(self.y_mask) 

    def _init_mask_generators(self, ParamsDict):
        if self.maskType in ["FreePhase", "FreePhaseTransmision"]:
            self.phaseMaskGenerator = FreeMaskGenerator(isPhaseMask=True).to(self.device)
        if self.maskType in ["FreeTransmision", "FreePhaseTransmision"]:
            self.transmisionMaskGenerator = FreeMaskGenerator(isPhaseMask=False).to(self.device)
        if self.maskType in ["Pyramid", "Zernike", "BiOEdge"]:
            self.param = nn.Parameter(torch.tensor(ParamsDict['InitParam'], device = self.device))
        if self.maskType == "ModalMask":
            self.phaseMaskGenerator = ModalMaskGeneration(self.N, self.device).to(self.device)
        if self.maskType in ["FullyFreePhase"]:
            self.phaseMask = nn.Parameter(torch.randn(self.N, self.N, device = self.device, dtype = torch.float32))
        if self.maskType in ["FullyFreeTransmision"]:
            self.transmisionMaskGenerator = nn.Parameter(0.00001*torch.randn(2, self.N, self.N, device = self.device, dtype = torch.float32))
        if self.maskType in ["Papyrus"]:
            self.mainSlope = nn.Parameter(torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32))
            self.maskShifts = nn.Parameter(torch.ones(4, 2, device = self.device, dtype = torch.float32))
            self.rooftop = nn.Parameter(torch.tensor(-2.2999, device=self.device, dtype=torch.float32))
            self.coordinatesRotation = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.float32))
            
        if self.maskType not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.maskType}")


    def update_masks(self):
        if self.maskType == "Pyramid":
            self.phaseMask = self.BuildPyramidMask()


        if self.maskType == "Zernike":
            self.phaseMask = self.BuildZernikeMask()

        
        if self.maskType == "BiOEdge":
            self.transmisionMask = self.BuildBiOEdgeMask()
            
        if self.maskType == "Papyrus":
            self.phaseMask = self.BuildPapyrusPyramidMask()
            
        
        if self.maskType == "FullyFreeTransmision":
            self.transmisionMask = self.DoubleTransmisionMask(torch.sigmoid(self.transmisionMaskGenerator[0]),
                                                              torch.sigmoid(self.transmisionMaskGenerator[1]))

        if self.maskType in ["FreePhase", "FreePhaseTransmision"]:
            self.phaseMask = self.phaseMaskGenerator(self.UV)
            self.phaseMask = self._remove_tip_tilt(self.phaseMask).view(self.N, self.N)


        if self.maskType in ["FreeTransmision", "FreePhaseTransmision"]:
            self.transmisionMask = self.transmisionMaskGenerator(self.UV).view(self.N, self.N)
                
        if self.maskType == "ModalMask":
            self.phaseMask = self.phaseMaskGenerator()
            
        if self.maskType not in mask_types_list:
            raise ValueError(f"Unsupported mask type: {self.maskType}")
             
        self.WFS.SetMask(phaseMask=self.phaseMask, transmisionMask=self.transmisionMask)
        

    def _remove_tip_tilt(self, mask):
        # Least-squares fit to remove linear plane (tip/tilt)
        coeffs = torch.linalg.lstsq(self.UV[self.circ_mask], mask[self.circ_mask]).solution
        tilt_plane = self.UV @ coeffs
        return mask - tilt_plane
    
    
    def GetPupilCenter(self):
        sign_tensor = torch.tensor([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]], device = self.device)
        frame_center = torch.ones(4,2, device = self.device) * self.N/2
        pupil_center = frame_center + sign_tensor * self.maskShifts * self.N/4
        pupil_center = torch.round(pupil_center).to(dtype = torch.int)
        return pupil_center


    def BuildZernikeMask(self):
       diameter_in_pixels = self.param[0] * self.sampling
       
       # this line is not differentiable I use a tanh function to model the mask
       #zernike_mask = self.param[1] * (rho < diameter_in_pixels / 2.)
       
       slope = self.param[2]
      
       ring_mask = torch.tanh(slope*( diameter_in_pixels/2. -self.rho_mask))/2
       annular = ring_mask+0.5

       zernike_mask = self.param[1] * annular
       
       return zernike_mask
   
    def BuildPyramidMask(self):
        pyramid_mask = (self.abs_x_mask * self.param[0] + self.abs_y_mask * self.param[1])
        
        return pyramid_mask
    
    
    def BuildPapyrusPyramidMask(self):
        
        rooftop_in_pixels = self.rooftop * self.sampling / np.sqrt(2)
        
        P1 = self.x_mask * self.maskShifts[0,0] + self.y_mask * self.maskShifts[0,1] + rooftop_in_pixels
        P2 = -self.x_mask * self.maskShifts[1,0] + self.y_mask * self.maskShifts[1,1]
        P3 = -self.x_mask * self.maskShifts[2,0] - self.y_mask * self.maskShifts[2,1] + rooftop_in_pixels
        P4 = self.x_mask * self.maskShifts[3,0] - self.y_mask * self.maskShifts[3,1]
        
        stacked = torch.stack([P1, P2, P3, P4])  # shape: (4, H, W)
        

        F = torch.max(stacked * self.mainSlope, dim=0).values  # shape (H, W)
        
        # F = P1 * mask_P1 + (P2 - rooftop) * mask_P2 + P3 * mask_P3 + (P4 - rooftop) * mask_P4
        
        #upScale = torch.nn.functional.interpolate(self.fieldDistortion.unsqueeze(0).unsqueeze(0), (self.N,self.N), mode = 'bilinear').squeeze()
        
        #distortion = self.fieldDistortion(self.UV).view(self.N, self.N)
        
        return F #, mask_P1, mask_P2, mask_P3, mask_P4
    
    def BuildBiOEdgeMask(self):
        
        return self.DoubleTransmisionMask(self.linear_ramp(self.x_mask, self.param[0]),
                                          self.linear_ramp(self.y_mask, self.param[0]))
    
    def DoubleTransmisionMask(self, mask_x, mask_y):
        mask = torch.zeros(1,4,self.N,self.N, device = self.device, dtype = torch.float32)
        
        m0 = mask_x
        m1 = 1 - m0
        
        m2 = mask_y
        m3 = 1 - m2
        
        mask[0,0] = m0
        mask[0,1] = m1
        mask[0,2] = m2
        mask[0,3] = m3
        
        return torch.sqrt(mask)
        
        
    def linear_ramp(self, x, delta):
        """
        x: input tensor
        a: start of linear ramp
        b: end of linear ramp
        """
        return torch.clamp((x + delta) / (2 * delta), min=0.0, max=1.0)
        


class ScaledTanh(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1)


class FreeMaskGenerator(nn.Module):
    """
    Neural network module that generates a phase or transmission mask 
    based on 2D input coordinates (u, v).

    Args:
        hidden_size (int): Number of hidden units in the fully connected layers.
        isPhaseMask (bool): Whether the mask should represent a phase mask (True) 
                            or a transmission mask (False).
    """
    def __init__(self, hidden_size=128, isPhaseMask = True):
        super().__init__()
        
        self.isPhaseMask = isPhaseMask
        
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),  # Input: (u, v)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Output: Mask value
        )

        # Apply custom weight initialization
        self.apply(self._init_weights)
        
        # If it is a transmision mask constrain to 0-1
        self.transmisionSigmoid = nn.Sigmoid()
        self.transmissionTanh = ScaledTanh()
        

    def _init_weights(self, module):
        """
        Applies custom initialization to the network weights.
        Weights are drawn from a normal distribution and biases are set to a constant.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.1)  # Normal distribution
            nn.init.constant_(module.bias, 0.1)  # Set bias to zero

    def forward(self, uv_coords):
        """
        Evaluates the mask for given UV coordinates.

        Args:
            uv_coords (Tensor): Tensor of shape (N, 2) containing normalized coordinates.

        Returns:
            Tensor: Output mask values (phase or transmission).
        """
        x = self.net(uv_coords) 
        if self.isPhaseMask:
            return x
    
        # x = self.transmisionSigmoid(x)
        x = self.transmissionTanh(x)
        return x


class ModalMaskGeneration(nn.Module):
    
    def __init__(self, maskResolution, device, NumberOfModes = 30):
        super().__init__()
        
        self.coefs = nn.Parameter(torch.randn(size=(NumberOfModes, 1, 1), device = device) / 100., requires_grad=True)
        
        self.modes = ZernikeFullView(maskResolution, range(3, NumberOfModes + 3)).to(device = device)
        
    def forward(self):
        return torch.sum(self.modes * self.coefs, dim = 0)

        

class MaskVisualizator:
    """
    Utility class for visualizing dynamically generated phase and/or transmission masks
    from an End-to-End Wavefront Sensor model.

    Args:
        E2E_WFS: The model containing the mask information and type.
    """
    def __init__(self, E2E_WFS, loss):
        self.E2E_WFS = E2E_WFS
        self.loss = loss
        
        
    def SetCanvas(self):
        """
        Initializes the matplotlib canvas and image for mask visualization
        based on the type of mask being used.
        """
        if self.E2E_WFS.maskType in ["FreePhase", "ModalMask", "FullyFreePhase"]:
            self.fig, self.ax = plt.subplots(1,3, figsize = (21, 5))
            self.img = self.ax[2].imshow(self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy())
            self.fig.colorbar(self.img)
            self.ax[2].set_title("Phase mask")
            
       
        
        elif self.E2E_WFS.maskType == "FreeTransmision":
            self.fig, self.ax = plt.subplots(1,3, figsize = (21, 5))
            self.img = self.ax[2].imshow(self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy())
            self.fig.colorbar(self.img)
            self.ax[2].set_title("Transmision mask")
            
            
        elif self.E2E_WFS.maskType == "FreePhaseTransmision":
            self.fig, self.ax = plt.subplots(1, 5, figsize = (21, 4))
            self.img1 = self.ax[2].imshow(self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy())
            self.fig.colorbar(self.img1)
            self.img2 = self.ax[3].imshow(self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy())
            self.fig.colorbar(self.img2)
            self.img2.set_clim(0, 1)
            
            self.ax[2].set_title("Phase mask")
            self.ax[3].set_title("Transmision mask")
            
        else:
            self.fig, self.ax = plt.subplots(1,2, figsize = (14, 5))
            
        plt.pause(0.3)


        self.lossPlot, = self.ax[0].plot(self.loss)
        self.reconstructionPlotTheoretical, = self.ax[1].plot(self.loss)
        self.reconstructionPlotEstimated, = self.ax[1].plot(self.loss)
        self.ax[0].set_title("Loss Evolution")
        self.ax[1].set_title("Sample Reconstruction")
        

            
    
    def show(self):
        """
        Updates and redraws the mask image(s) based on current model parameters.
        Called regularly during training to reflect mask updates.
        """
        if self.E2E_WFS.maskType in ["FreePhase", "ModalMask", "FullyFreePhase"]:
            self.img.set_data(self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy())
            self.img.set_clim(vmin=np.min(self.img.get_array()), vmax=np.max(self.img.get_array()))

 
        elif self.E2E_WFS.maskType == "FreeTransmision":
            self.img.set_data(self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy())
            self.img.set_clim(vmin=np.min(self.img.get_array()), vmax=np.max(self.img.get_array()))
            
            
        elif self.E2E_WFS.maskType == "FreePhaseTransmision":
            self.img1.set_data(self.E2E_WFS.maskManager.phaseMask.cpu().detach().numpy())
            self.img1.set_clim(vmin=np.min(self.img1.get_array()), vmax=np.max(self.img1.get_array()))
            
            self.img2.set_data(self.E2E_WFS.maskManager.transmisionMask.cpu().detach().numpy())
        
        plt.pause(0.1)
        
    def update_plots(self, zernikeTeo, zernikeEst):
        smooth_loss = np.convolve(self.loss, np.ones(100)/100, 'valid')
        self.lossPlot.set_xdata(np.arange(len(smooth_loss)))
        self.lossPlot.set_ydata(smooth_loss)
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        
        self.reconstructionPlotTheoretical.set_xdata(np.arange(len(zernikeTeo[0])))
        self.reconstructionPlotTheoretical.set_ydata(zernikeTeo[0].cpu().detach())
        self.reconstructionPlotEstimated.set_xdata(np.arange(len(zernikeEst[0])))
        self.reconstructionPlotEstimated.set_ydata(zernikeEst[0].cpu().detach())
        self.ax[1].relim()
        self.ax[1].autoscale_view()


def trainMask (maskGenerator, uv_coords, mask, loss, TrainRunNb, optimizer, device = 'cuda'):
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
    
    img = ax.imshow(maskGenerator(uv_coords).view(N,N).cpu().detach().numpy())
    fig.colorbar(img)
    plt.show()
    
    writer = imageio.get_writer(gif_path, mode="i", fps = 10)
    
    maskGenerator.train()

    for u in range(0,TrainRunNb) :
        

        optimizer.zero_grad()

        output = maskGenerator(uv_coords).view(N, N)  
        
        l = loss(output,mask.to(output.dtype))
    
        l. backward()
        
        optimizer.step()
        

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
   
    N = 140
    
    # Setting the loss function
    loss = torch.nn.MSELoss()
    
    # Initialisation of the system 
    maskGenerator = FreeMaskGenerator().to(device)
    
    u = torch.linspace(-N//2, N // 2 - 1, N) / (N/2)  # Normalized frequency range
    U, V = torch.meshgrid(u, u, indexing="xy")  # Create the full grid
    
    pyr_mask = (np.pi / 2 * (torch.abs(U) + torch.abs(V)) * N/2).to(device)
    zernike_mask = np.pi/2 * (torch.sqrt(U**2 + V**2) < 4/N).to(device)
    random_mask = torch.randn_like(zernike_mask)
    
    uv_coords = torch.stack([U.flatten(), V.flatten()], dim=1).to(device)

    # Flatten and stack into (N^2, 2) shape
    
   
    
    # Optimization parameters (learning rate lr and nb of runs)
   
   
    optimizer = torch.optim.Adam(maskGenerator.parameters(),0.001)

    
    a = time.time()
    train_loss = trainMask(maskGenerator, uv_coords, pyr_mask,loss,3000,optimizer,device)
    b = time.time() - a 
    
    torch.save({
        'Phase_Mask_state_dict': maskGenerator.state_dict(),
        'optimizer_o_state_dict': optimizer.state_dict()
        }, mask_path)
    
