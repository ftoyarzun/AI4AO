import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import numpy as np
from .TorchPropagator import WFS
from .Utils import MakePupil

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
        transmisionMask = MakePupil(N, self.device)
        transmisionMask = transmisionMask.repeat(1, self.number_of_masks, 1, 1)
        phaseMask= phaseMask * self.depths.view(1, self.number_of_masks, 1, 1)
        self.MakeMTFMatrices(self.diameters[0])

        frame_center = torch.ones((self.number_of_masks, 2), device=self.device, dtype=torch.float32) * self.Npix // 2
        pupil_center = (frame_center + self.positions / 2 / np.pi * self.Npix)
        self.pupil_centers = np.round(pupil_center.detach().cpu().numpy()).astype(np.int32)

        # Unrounded (row, col) pixel offset from frame center, used by
        # WFS.ShiftPupilImages for a sub-pixel, both-axes pupil-image shift --
        # column 0 is the row (dims=-2) component, column 1 the col (dims=-1)
        # component, matching BuildZernikeMaskFFT's positions/coords pairing.
        pupil_shift = pupil_center - frame_center
        self.pupil_shift_row = pupil_shift[:, 0]
        self.pupil_shift_col = pupil_shift[:, 1]
        self.BuildShiftGrid(self.pupil_shift_row, self.pupil_shift_col)

        return phaseMask, transmisionMask
    
    def forward(self, opd, pupil = None):
        return self.Propagator(opd, pupil)
    

