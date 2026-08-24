import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
from .TorchPropagator import WFS

class PyramidWFS(WFS):
    def __init__(self, ParamsDict, device):
        super().__init__(ParamsDict, device)

        self.mainSlope = nn.Parameter(torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32))
        self.maskShifts = nn.Parameter(torch.ones(4, 2, device=self.device, dtype=torch.float32))
        self.rooftop = nn.Parameter(torch.tensor(0, device=self.device, dtype=torch.float32))

        self.modulation = ParamsDict["Modulation"]

        self.BuildMask()

    def BuildMask(self):

        rooftop_in_pixels = self.rooftop * self.sampling / np.sqrt(2)

        P1 = (self.x_mask + rooftop_in_pixels / 2) * self.maskShifts[0, 0] + (
            self.y_mask + rooftop_in_pixels / 2
        ) * self.maskShifts[0, 1]
        P2 = -self.x_mask * self.maskShifts[1, 0] + self.y_mask * self.maskShifts[1, 1]
        P3 = (
            -(self.x_mask - rooftop_in_pixels / 2) * self.maskShifts[2, 0]
            - (self.y_mask - rooftop_in_pixels / 2) * self.maskShifts[2, 1]
        )
        P4 = self.x_mask * self.maskShifts[3, 0] - self.y_mask * self.maskShifts[3, 1]

        stacked = torch.stack([P1, P2, P3, P4])  # shape: (4, H, W)

        F = torch.max(stacked * self.mainSlope, dim=0).values  # shape (H, W)

        self.pupil_centers = self.GetPupilCenter()

        self.SetMask(phaseMask=F)
    
    def GetPupilCenter(self):
        sign_tensor = -torch.tensor(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=self.device
        )
        frame_center = torch.ones(4, 2, device=self.device) * self.Npix / 2
        pupil_center = frame_center + sign_tensor * self.maskShifts * self.mainSlope * self.Npix / 4 / (torch.pi/2)
        pupil_center = torch.round(pupil_center).to(dtype=torch.int).cpu().detach().numpy()
        return pupil_center
    
    def PropagateField(self, uin, uin_padded):
        self.FFTPropagator(uin_padded)

    def BuildPrismMask(self, pupil_proportion, Nsamples = 5):
        self.BuildMask()
        displacement_size_in_pix = self.Nres * pupil_proportion / self.sampling
        displacement_array = torch.linspace(
            -displacement_size_in_pix/2, displacement_size_in_pix/2, Nsamples, 
            device = self.device, dtype = torch.float32
            ).view(Nsamples,1,1)

        standard_pupil_displacement_in_pix = self.mainSlope / (2 * torch.pi) * self.Npix
        samples_pupil_positions_array = displacement_array + standard_pupil_displacement_in_pix

        displacement_factor = samples_pupil_positions_array / self.mainSlope / self.Npix * (2 * torch.pi)

        mask = torch.clone(self.phaseMask).repeat((Nsamples,1,1))

        mask *= displacement_factor

        self.SetMask(phaseMask=mask)


    
   