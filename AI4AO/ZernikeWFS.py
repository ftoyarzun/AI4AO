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

        self.initialized = True
    
    def BuildMask(self):
        if self.use_MTF is False:
            self.phaseMask = self.BuildZernikeMaskFFT()
            self.SetMask(phaseMask=self.phaseMask)

        if self.use_MTF is True:
            self.phaseMask, self.transmisionMask = self.BuildZernikeMaskMFT()
            self.SetMask(phaseMask=self.phaseMask, transmisionMask=self.transmisionMask)

    
    def BuildZernikeMaskFFT(self):
        """Builds the (Nwavelength, Nmask, H, W) FFT-path phase mask.

        The dot diameter (in lambda_c/D) and depth (radians at lambda_c) are
        given at the band's central wavelength (see WFS.ChromaticRatio). The dot
        is a fixed angle, so at each wavelength its diameter spans
        sampling * lambda_c / lambda pixels. It is a fixed step in optical path,
        so its phase depth scales as lambda_c / lambda (glass dispersion is
        ignored). The tilt ramps that separate the pupil images are linear, and
        therefore scale-invariant and achromatic.
        """
        ratio = self.ChromaticRatio().view(-1, 1, 1, 1)  # (Nwavelength, 1, 1, 1), exactly 1 for one wavelength

        coords = torch.stack([-self.x_mask, -self.y_mask], dim=0)
        phaseMask = torch.einsum('ck,kwh->cwh', self.positions, coords).unsqueeze(0)  # (1, Nmask, H, W)

        frame_center = torch.ones(self.number_of_masks, 2, device=self.device) * self.Npix / 2
        pupil_center = frame_center + self.positions / 2 / torch.pi * self.Npix
        self.pupil_centers = torch.round(pupil_center).to(dtype=torch.int).cpu().numpy()

        slope = 10
        diameters_in_pixels = self.diameters.reshape(1, -1, 1, 1) * (self.sampling * ratio)  # (Nwavelength, 1, 1, 1)

        ring_mask = (torch.tanh(slope * (diameters_in_pixels/ 2.0 - self.rho_mask))/ 2)
        annular = ring_mask + 0.5  # (Nwavelength, 1, H, W)

        zernike_mask = (self.depths.reshape(1, -1, 1, 1) * ratio) * annular  # (Nwavelength, Nmask, H, W)

        return phaseMask + zernike_mask
    
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
    

