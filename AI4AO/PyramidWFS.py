import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
from .TorchPropagator import WFS

class PyramidWFS(WFS):
    def __init__(self, ParamsDict, device):
        super().__init__(ParamsDict, device)

        self.initialized = False

        self.mainSlope = nn.Parameter(torch.tensor(torch.pi / 2, device=self.device, dtype=torch.float32))
        self.maskShifts = nn.Parameter(torch.ones(4, 2, device=self.device, dtype=torch.float32))
        self.rooftop = nn.Parameter(torch.tensor(0, device=self.device, dtype=torch.float32))

        self.modulation = ParamsDict["Modulation"]

        self.BuildMask()

        self.initialized = True

    def BuildMask(self):
        if self.modulation == 0:
            F = self.PyramidMask()
        else:
            nSteps = min(32, max(round(6.28 * self.modulation / 4) * 4, 8))
            steps = torch.linspace(0,2*torch.pi,nSteps, device=self.device, dtype=torch.float32)
            x = self.modulation * self.sampling * torch.cos(steps)
            y = self.modulation * self.sampling * torch.sin(steps)
            F = self.PyramidMask(x_offset=x, y_offset=y)
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

    def PyramidMask(self, x_offset=0, y_offset=0):
        # scalar offsets -> single (H, W) mask; array-like offsets of length C -> (C, H, W), one mask per offset pair
        x_offset = torch.as_tensor(x_offset, device=self.device, dtype=torch.float32)
        y_offset = torch.as_tensor(y_offset, device=self.device, dtype=torch.float32)
        squeeze_output = x_offset.dim() == 0 and y_offset.dim() == 0
        x_offset, y_offset = torch.broadcast_tensors(x_offset.reshape(-1), y_offset.reshape(-1))

        rooftop_in_pixels = self.rooftop * self.sampling / np.sqrt(2)

        x = self.x_mask.unsqueeze(0) + x_offset.view(-1, 1, 1)  # (C, H, W)
        y = self.y_mask.unsqueeze(0) + y_offset.view(-1, 1, 1)  # (C, H, W)

        P1 = (x + rooftop_in_pixels / 2) * self.maskShifts[0, 0] + (
            y + rooftop_in_pixels / 2
        ) * self.maskShifts[0, 1]
        P2 = -x * self.maskShifts[1, 0] + y * self.maskShifts[1, 1]
        P3 = (
            -(x - rooftop_in_pixels / 2) * self.maskShifts[2, 0]
            - (y - rooftop_in_pixels / 2) * self.maskShifts[2, 1]
        )
        P4 = x * self.maskShifts[3, 0] - y * self.maskShifts[3, 1]

        stacked = torch.stack([P1, P2, P3, P4])  # shape: (4, C, H, W)

        F = torch.max(stacked * self.mainSlope, dim=0).values  # shape (C, H, W)

        if squeeze_output:
            F = F.squeeze(0)

        return F

    @property
    def modulation(self):
        return self._modulation
    @modulation.setter
    def modulation(self, value):
        with torch.no_grad():
            self._modulation = value
            if self.initialized:
                self.BuildMask()
                self.BuildReferenceIntensity()

    
   