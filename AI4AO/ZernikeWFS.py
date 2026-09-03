import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
from .TorchPropagator import WFS

class ZernikeWFS(WFS):
    def __init__(self, ParamsDict, device):
        super().__init__(ParamsDict, device)

        self.MTF_focal_upscale = ParamsDict["MTF_upscale"]
        self.use_MTF = ParamsDict["Use_MTF"]
        self.maskType = ParamsDict["MaskType"]
        self.modulation = 0

        if self.maskType.lower() in ["doublezernike", "vzwfs", "v-zwfs", "vectorzwfs", "vector-zwfs"]:
            self.depths = nn.Parameter(torch.tensor([-torch.pi * 0.2, torch.pi * 0.5], device=self.device, dtype=torch.float32))
            self.diameters = nn.Parameter(torch.tensor([2.], device=self.device, dtype=torch.float32))
            self.positions = nn.Parameter(torch.tensor([[-torch.pi * 0.5, 0], [torch.pi * 0.5, 0.0]], device=self.device, dtype=torch.float32))
            self.number_of_masks = 2

        if self.maskType.lower() in ["zernike", "zwfs"]:
            self.depths = nn.Parameter(torch.tensor([[torch.pi * 0.5]], device=self.device, dtype=torch.float32))
            self.diameters = nn.Parameter(torch.tensor([[2.]], device=self.device, dtype=torch.float32))
            self.positions = nn.Parameter(torch.tensor([[0.0, 0.0]], device=self.device, dtype=torch.float32))
            self.number_of_masks = 1

        self.BuildMask()
    
    def BuildMask(self):
        if self.use_MTF is False:
            self.phaseMask = self.BuildZernikeMaskFFT()
            self.SetMask(phaseMask=self.phaseMask)

        if self.use_MTF is True:
            self.phaseMask, self.transmisionMask = self.BuildZernikeMaskMFT()
            self.SetMask(phaseMask=self.phaseMask, transmisionMask=self.transmisionMask)

    
    def BuildZernikeMaskFFT(self):

        coords = torch.stack([-self.x_mask, -self.y_mask], dim=0)
        phaseMask = torch.einsum('ck,kwh->cwh', self.positions, coords).unsqueeze(0)

        frame_center = torch.ones(self.number_of_masks, 2, device=self.device) * self.Npix / 2
        pupil_center = frame_center + self.positions / 2 / torch.pi * self.Npix
        self.pupil_centers = torch.round(pupil_center).to(dtype=torch.int).cpu().numpy()

        slope = 10
        diameters_in_pixels = self.diameters * self.sampling
        diameters_in_pixels = diameters_in_pixels.unsqueeze(1).unsqueeze(1)

        ring_mask = (torch.tanh(slope * (diameters_in_pixels/ 2.0 - self.rho_mask.unsqueeze(0)))/ 2)
        annular = ring_mask + 0.5

        zernike_mask = self.depths.unsqueeze(1).unsqueeze(1) * annular

        phaseMask[0] = phaseMask[0] + zernike_mask

        return phaseMask
    
    def BuildZernikeMaskMFT(self):
        N = int(self.sampling * self.MTF_focal_upscale * self.diameters[0])
        phaseMask = torch.ones(1, self.number_of_masks, 1, 1, device=self.device, dtype=torch.float32)
        transmisionMask = self.make_pupil(self.sampling * self.MTF_focal_upscale * self.diameters / 2, N)
        transmisionMask = transmisionMask.repeat(1, self.number_of_masks, 1, 1)
        phaseMask= phaseMask * self.depths.view(1, self.number_of_masks, 1, 1)
        self.MakeMTFMatrices(self.diameters[0])

        frame_center = np.ones((self.number_of_masks, 2)) * self.Npix // 2
        pupil_center = (frame_center - self.positions.detach().cpu().numpy() / 2 / np.pi * self.Npix)
        self.pupil_centers = np.round(pupil_center).astype(np.int32)

        self.pupil_shifts = (self.pupil_centers - frame_center)[:, 0].astype(np.int32)

        return phaseMask, transmisionMask

    def make_pupil(self, Rpx, nPx):
        """
        Generate a circular pupil mask.

        Parameters
        ----------
        Rpx : float
            Radius of the pupil in pixels.
        nPx : int
            Size of the square output array (number of pixels per side).

        Returns
        -------
        pupil : ndarray of shape (nPx, nPx)
            Binary circular mask with value 1 inside the pupil and 0 outside.
        """
        # Create grid of coordinates
        x = torch.arange(nPx, device=self.device, dtype=torch.float32) - (nPx - 1) / 2
        x, y = torch.meshgrid(x, x)

        # Circular mask
        pupil = (x**2 + y**2) <= Rpx**2
        return pupil
    
    def forward(self, phase, pupil = None):
        return self.Propagator(phase,pupil)
    

